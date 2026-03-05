"""Wrapper around the ScyllaDB vector database for VectorDB benchmarks.

Uses the new ``scylla`` Python-rs driver (async, Rust-backed) instead of the
legacy ``cassandra-driver`` / ``scylla-driver``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Final

from scylla.enums import Consistency
from scylla.execution_profile import ExecutionProfile
from scylla.session_builder import SessionBuilder

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import ScyllaDBIndexScope

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from scylla.session import Session
    from scylla.statement import PreparedStatement

    from .config import ScyllaDBIndexConfig

__all__ = ["ScyllaDB"]

log = logging.getLogger(__name__)

_INDEX_POLL_INTERVAL_SEC: Final[float] = 1.0
_INDEX_BUILD_TIMEOUT_SEC: Final[float] = 3600.0

# Default CQL native transport port
_DEFAULT_PORT: Final[int] = 9042


def _run(coro):
    """Run an async coroutine from synchronous code.

    Tries to use the current running loop if available (and schedules via
    a thread), otherwise falls back to ``asyncio.run()``.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing event loop (e.g. Jupyter, Streamlit).
        # Cannot call asyncio.run() because it would try to create a new
        # event loop on this thread, which is forbidden while one is already
        # running.  Instead, spawn a *single* worker thread that has no
        # running loop and call asyncio.run(coro) there.
        #
        # max_workers=1 is intentional: we only ever submit one task, so the
        # pool exists solely to escape the current thread's event loop.  All
        # real concurrency (e.g. asyncio.gather inside the coroutine) happens
        # within the fresh event loop that asyncio.run() creates on that
        # single thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class ScyllaDB(VectorDB):
    """ScyllaDB client for vector database operations.

    Manages connection lifecycle, schema creation, data ingestion,
    and ANN search against a ScyllaDB cluster with vector-search support.

    This implementation uses the new async Python-rs driver from
    ``scylladb-zpp-2025-python-rs-driver/python-rs-driver``.
    """

    supported_filter_types: ClassVar[list[FilterOp]] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    # -- construction --------------------------------------------------------

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ScyllaDBIndexConfig,
        collection_name: str = "vdb_bench_collection",
        id_col_name: str = "id",
        label_col_name: str = "filtering_label",
        vector_field: str = "vector",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ) -> None:
        self.name = "ScyllaDB"
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.vector_field = vector_field
        self.with_scalar_labels = with_scalar_labels

        # Mutable state -- set by init() / prepare_filter()
        self.session: Session | None = None
        self.prepared_insert: PreparedStatement | None = None
        self.prepared_lookup: PreparedStatement | None = None
        self._filter_params: tuple[object, ...] = ()

        # Persistent event loop for the init() context – avoids the cost
        # of creating / tearing down a loop on every _run() call.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

        log.info(
            "%s using python-rs (Rust-backed) driver.",
            self.name,
        )
        log.info("%s index params: %s", self.name, self.case_config.index_param())

        async def _setup():
            session = await self._connect()
            try:
                if drop_old:
                    log.info("%s dropping old table: %s", self.name, self.table_name)
                    await session.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                    await self._create_table(session)
                    if not self.case_config.create_index_after_upload:
                        await self._create_index(session)
            finally:
                # The python-rs driver session has no explicit shutdown;
                # just let it go out of scope.
                pass

        _run(_setup())

    # -- authentication helpers -----------------------------------------------

    @staticmethod
    def _read_credentials(env_path: str = ".env") -> tuple[str | None, str | None]:
        """Read ScyllaDB credentials from environment.

        Returns ``(username, password)``; either or both may be ``None``.
        """
        import environs  # optional dependency -- imported lazily

        env = environs.Env()
        env.read_env(path=env_path, recurse=False)
        username: str | None = env("SCYLLADB_USERNAME", default=None)
        password: str | None = env("SCYLLADB_PASSWORD", default=None)

        if (username is None) != (password is None):
            log.warning(
                "Only one of SCYLLADB_USERNAME / SCYLLADB_PASSWORD is set; "
                "authentication may fail."
            )
        return username, password

    # -- connection helpers ---------------------------------------------------

    def _build_session_builder(self, contact_points: list[str]) -> SessionBuilder:
        """Create a :class:`SessionBuilder` for the configured cluster."""
        # The python-rs driver's SessionBuilder does not support auth yet.
        # Credentials are read but only logged as a warning for now.
        username, password = self._read_credentials()
        if username and password:
            log.warning(
                "%s: python-rs driver does not yet support authentication; "
                "credentials are ignored.",
                self.name,
            )
        # Default to Consistency.One for all statements – sufficient for
        # benchmarking and avoids the latency of LocalQuorum.
        profile = ExecutionProfile(consistency=Consistency.One)
        return SessionBuilder(contact_points, _DEFAULT_PORT, execution_profile=profile)

    async def _connect(self, keyspace: str | None = None) -> Session:
        """Open a connection returning the async Session.

        If *keyspace* is ``None`` the configured keyspace is created (if
        needed) and selected on the session automatically.
        """
        uri = self.db_config["cluster_uris"]
        ks = keyspace or self.db_config["keyspace"]

        builder = self._build_session_builder(uri)
        log.info("%s connecting to cluster at %s", self.name, uri)
        session = await builder.connect()

        if keyspace is None:
            await self._create_keyspace(session, ks)
        await session.execute(f"USE {ks}")
        return session

    def _ensure_session(self) -> Session:
        """Return the active session or raise if ``init()`` was not called."""
        if self.session is None:
            msg = (
                f"{self.name}: no active session -- "
                "wrap operations inside `with self.init():`"
            )
            raise RuntimeError(msg)
        return self.session

    # -- schema management ---------------------------------------------------

    @property
    def _use_local_index(self) -> bool:
        """Whether to use a local (partition-level) secondary index."""
        return (
            self.with_scalar_labels
            and self.case_config.index_scope == ScyllaDBIndexScope.LOCAL
        )

    async def _create_keyspace(self, session: Session, keyspace: str) -> None:
        """Create keyspace if it does not exist."""
        log.info("%s creating keyspace: %s", self.name, keyspace)
        replication_factor = self.db_config.get("replication_factor", 1)
        # Tablets require NetworkTopologyStrategy; SimpleStrategy is not supported.
        strategy = "NetworkTopologyStrategy"
        await session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': '{strategy}', "
            f"'replication_factor': '{replication_factor}'}} "
            f"AND tablets = {{'enabled': 'true'}}"
        )

    async def _create_table(self, session: Session) -> None:
        """Create table for vector storage."""
        if self._use_local_index:
            pk = f"PRIMARY KEY ({self.label_col_name}, {self.id_col_name})"
        elif self.with_scalar_labels:
            pk = f"PRIMARY KEY ({self.id_col_name}, {self.label_col_name})"
        else:
            pk = f"PRIMARY KEY ({self.id_col_name})"
        label_col = (
            f"{self.label_col_name} text,"
            if self.with_scalar_labels
            else ""
        )
        create_table_cql = (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            f"  {self.id_col_name} int,"
            f"  {label_col}"
            f"  {self.vector_field} vector<float, {self.dim}>,"
            f"  {pk}"
            f")"
        )
        await session.execute(create_table_cql)
        log.info("%s created table: %s", self.name, self.table_name)

    async def _create_index(self, session: Session) -> None:
        """Create vector search index on the table."""
        if self._use_local_index:
            target = f"(({self.label_col_name}), {self.vector_field})"
        else:
            target = f"({self.vector_field})"
        create_index_cql = (
            f"CREATE CUSTOM INDEX IF NOT EXISTS ON {self.table_name} "
            f"{target} USING 'vector_index' "
            f"WITH OPTIONS = {self.case_config.index_param()}"
        )
        await session.execute(create_index_cql)
        log.info("%s created index on: %s", self.name, self.table_name)

    # -- lifecycle (per-process) ---------------------------------------------

    # -- persistent event loop helpers ----------------------------------------

    def _start_loop(self) -> None:
        """Spin up a background thread running a persistent event loop.

        All async driver calls within an ``init()`` context are dispatched
        to this loop via :pymethod:`_run_async`, avoiding the overhead of
        ``asyncio.run()`` (which creates *and* destroys a loop each time).
        """
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="scylladb-event-loop",
        )
        self._loop_thread.start()

    def _stop_loop(self) -> None:
        """Shut down the persistent event loop and its thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10.0)
        self._loop = None
        self._loop_thread = None

    def _run_async(self, coro):
        """Schedule *coro* on the persistent loop, or fall back to ``_run``."""
        loop = self._loop
        if loop is not None and loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return _run(coro)

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Create and destroy connections to the database.

        Must be used as a context manager before calling any data or search
        operations::

            with db.init():
                db.insert_embeddings(...)
                db.search_embedding(...)
        """
        self._start_loop()

        async def _open():
            uri = self.db_config["cluster_uris"]
            keyspace = self.db_config["keyspace"]
            builder = self._build_session_builder(uri)
            session = await builder.connect()
            await session.execute(f"USE {keyspace}")
            return session

        self.session = self._run_async(_open())
        self._prepare_insert_statement()

        try:
            yield
        finally:
            self._reset_session_state()

    def _reset_session_state(self) -> None:
        """Clear all per-session state."""
        # The python-rs driver session has no explicit shutdown.
        self.session = None
        self.prepared_insert = None
        self.prepared_lookup = None
        self._filter_params = ()
        self._stop_loop()

    def _prepare_insert_statement(self) -> None:
        """Prepare the CQL INSERT statement for the current session."""
        session = self._ensure_session()

        if self.with_scalar_labels:
            columns = f"{self.id_col_name}, {self.vector_field}, {self.label_col_name}"
            placeholders = "?, ?, ?"
        else:
            columns = f"{self.id_col_name}, {self.vector_field}"
            placeholders = "?, ?"

        prepared = self._run_async(
            session.prepare(
                f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
            )
        )
        # Set consistency ONE on the prepared statement
        self.prepared_insert = prepared.with_consistency(Consistency.One)

    # -- data operations -----------------------------------------------------

    def need_normalize_cosine(self) -> bool:
        """Whether this database needs to normalize dataset to support COSINE."""
        return True

    def insert_embeddings(
        self,
        embeddings: Sequence[list[float]],
        metadata: Sequence[int],
        labels_data: Sequence[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings into ScyllaDB using asyncio.gather for concurrency.

        Instead of batch statements, each row is inserted as a separate
        async execute call, gathered concurrently via ``asyncio.gather``.

        Args:
            embeddings: Vectors to insert.
            metadata:   Integer keys (IDs) for each vector.
            labels_data: Optional string labels (only when scalar labels are enabled).

        Returns:
            Tuple of (inserted_count, error_or_None).
        """
        session = self._ensure_session()
        assert self.prepared_insert is not None, "prepared_insert not initialized"

        async def _insert_batch():
            if self.with_scalar_labels:
                if labels_data is None:
                    raise ValueError(
                        "labels_data is required when with_scalar_labels is True"
                    )
                coros = [
                    session.execute(self.prepared_insert, [key, embedding, label])
                    for key, embedding, label in zip(
                        metadata, embeddings, labels_data, strict=True
                    )
                ]
            else:
                coros = [
                    session.execute(self.prepared_insert, [key, embedding])
                    for key, embedding in zip(metadata, embeddings, strict=True)
                ]
            await asyncio.gather(*coros)

        try:
            self._run_async(_insert_batch())
        except Exception as e:
            log.warning("%s failed to insert data: %s", self.name, e)
            return 0, e
        return len(embeddings), None

    # -- search & filtering --------------------------------------------------

    def prepare_filter(self, filters: Filter) -> None:
        """Pre-prepare filter conditions to reduce redundancy during search.

        Filter values are bound via CQL prepared-statement parameters
        rather than interpolated into the query string.
        """
        session = self._ensure_session()

        if filters.type == FilterOp.NonFilter:
            where = ""
            allow_filtering = ""
            self._filter_params = ()
        elif filters.type == FilterOp.NumGE:
            where = f" WHERE {self.id_col_name} > ?"
            allow_filtering = " ALLOW FILTERING"
            self._filter_params = (filters.int_value,)
        elif filters.type == FilterOp.StrEqual:
            where = f" WHERE {self.label_col_name} = ?"
            allow_filtering = "" if self._use_local_index else " ALLOW FILTERING"
            self._filter_params = (filters.label_value,)
        else:
            msg = f"Unsupported filter for {self.name}: {filters}"
            raise ValueError(msg)

        prepared = self._run_async(
            session.prepare(
                f"SELECT {self.id_col_name} FROM {self.table_name}"
                f"{where} "
                f"ORDER BY {self.vector_field} ANN OF ? LIMIT ?"
                f"{allow_filtering}"
            )
        )
        self.prepared_lookup = prepared.with_consistency(Consistency.One)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        """Get *k* most similar embeddings to *query*.

        Args:
            query: Query embedding to look up documents similar to.
            k:     Number of most similar embeddings to return.

        Returns:
            List of *k* most similar embedding IDs.

        Raises:
            RuntimeError: If ``prepare_filter`` was not called first.
        """
        session = self._ensure_session()
        if self.prepared_lookup is None:
            msg = (
                f"{self.name}: prepared_lookup is not set -- "
                "call prepare_filter() before searching"
            )
            raise RuntimeError(msg)

        result = self._run_async(
            session.execute(
                self.prepared_lookup,
                list(self._filter_params) + [query, k],
            )
        )
        return [row[self.id_col_name] for row in result.iter_rows()] if result else []

    # -- optimisation --------------------------------------------------------

    def _wait_for_index_build(
        self,
        timeout: float = _INDEX_BUILD_TIMEOUT_SEC,
        poll_interval: float = _INDEX_POLL_INTERVAL_SEC,
    ) -> None:
        """Block until the ANN index is queryable.

        Args:
            timeout:       Maximum seconds to wait before raising ``TimeoutError``.
            poll_interval: Seconds between successive probe queries.
        """
        session = self._ensure_session()
        log.info("%s waiting for index build to complete ...", self.name)

        sample_vector = [0.0] * self.dim
        probe_cql = (
            f"SELECT * FROM {self.table_name} "
            f"ORDER BY {self.vector_field} ANN OF ? LIMIT 1"
        )
        # Prepare once to avoid repeated server-side parsing in the poll loop.
        prepared_probe = self._run_async(session.prepare(probe_cql))

        deadline = time.monotonic() + timeout
        while True:
            try:
                self._run_async(session.execute(prepared_probe, [sample_vector]))
            except Exception as e:
                if time.monotonic() >= deadline:
                    msg = f"{self.name}: index not ready after {timeout}s"
                    raise TimeoutError(msg) from e
                log.debug("%s index not ready yet: %s", self.name, e)
            else:
                log.info("%s index build completed.", self.name)
                return
            time.sleep(poll_interval)

    def optimize(self, data_size: int | None = None) -> None:
        """Create index (if deferred) and wait for it to be fully built before search benchmarks."""
        if self.case_config.create_index_after_upload:
            session = self._ensure_session()
            self._run_async(self._create_index(session))
        self._wait_for_index_build()
