"""Unit tests for the ScyllaDB VectorDB client (python-rs driver).

All driver interactions are mocked — no running ScyllaDB instance is required.
The ``scylla`` package does not need to be installed; the entire module
hierarchy is injected into ``sys.modules`` before the client module is loaded.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Inject fake ``scylla`` package into sys.modules BEFORE any imports that
# depend on it (the client module has ``from scylla.enums import …`` etc.)
# ---------------------------------------------------------------------------

_scylla_mod = ModuleType("scylla")
_scylla_enums = ModuleType("scylla.enums")
_scylla_session_builder = ModuleType("scylla.session_builder")
_scylla_session = ModuleType("scylla.session")
_scylla_statement = ModuleType("scylla.statement")

# Provide the symbols expected by scylladb.py at import time
_scylla_enums.Consistency = MagicMock(name="Consistency")
_scylla_enums.Consistency.One = MagicMock(name="Consistency.One")
_scylla_session_builder.SessionBuilder = MagicMock(name="SessionBuilder")
_scylla_session.Session = MagicMock(name="Session")
_scylla_statement.PreparedStatement = MagicMock(name="PreparedStatement")

for name, mod in {
    "scylla": _scylla_mod,
    "scylla.enums": _scylla_enums,
    "scylla.session_builder": _scylla_session_builder,
    "scylla.session": _scylla_session,
    "scylla.statement": _scylla_statement,
}.items():
    sys.modules.setdefault(name, mod)

# Force-reload the client module so it picks up the fakes
import vectordb_bench.backend.clients.scylladb.scylladb as _scylladb_module

importlib.reload(_scylladb_module)

from vectordb_bench.backend.clients.scylladb.scylladb import ScyllaDB, _run
from vectordb_bench.backend.clients.scylladb.config import (
    ScyllaDBIndexConfig,
    ScyllaDBIndexScope,
)
from vectordb_bench.backend.clients.api import MetricType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_config(
    keyspace: str = "test_ks",
    uris: list[str] | None = None,
    replication_factor: int = 1,
) -> dict:
    return {
        "keyspace": keyspace,
        "cluster_uris": uris or ["127.0.0.1"],
        "replication_factor": replication_factor,
    }


def _make_case_config(**overrides) -> ScyllaDBIndexConfig:
    defaults = dict(metric_type=MetricType.L2)
    defaults.update(overrides)
    return ScyllaDBIndexConfig(**defaults)


def _mock_prepared():
    """Return a mock PreparedStatement whose .with_consistency() returns itself."""
    prep = MagicMock(name="PreparedStatement")
    prep.with_consistency.return_value = prep
    return prep


def _mock_session(prepare_rv=None):
    """Return an AsyncMock Session with .execute() and .prepare() coroutines."""
    session = AsyncMock(name="Session")
    session.execute = AsyncMock(return_value=MagicMock())
    if prepare_rv is None:
        prepare_rv = _mock_prepared()
    session.prepare = AsyncMock(return_value=prepare_rv)
    return session


def _mock_session_builder(session):
    """Return a MagicMock SessionBuilder whose .connect() resolves to *session*."""
    builder = MagicMock(name="SessionBuilder")
    builder.connect = AsyncMock(return_value=session)
    return builder


def _build_scylladb(dim=4, drop_old=False, with_scalar_labels=False, **extra):
    """Construct a ScyllaDB instance with all driver calls mocked.

    Patches SessionBuilder so the constructor's _connect() uses our mock
    session, and patches _read_credentials to avoid .env lookups.
    """
    session = _mock_session()
    builder = _mock_session_builder(session)
    case_config = _make_case_config(**extra)

    with (
        patch.object(
            _scylladb_module,
            "SessionBuilder",
            return_value=builder,
        ),
        patch.object(
            ScyllaDB,
            "_read_credentials",
            return_value=(None, None),
        ),
    ):
        db = ScyllaDB(
            dim=dim,
            db_config=_make_db_config(),
            db_case_config=case_config,
            drop_old=drop_old,
            with_scalar_labels=with_scalar_labels,
        )

    return db, session, builder


# ===========================================================================
# Tests
# ===========================================================================


class TestScyllaDBConstruction:
    """Tests for __init__ / schema setup."""

    def test_basic_construction_no_drop(self):
        """Constructor with drop_old=False must NOT issue DROP/CREATE statements."""
        db, session, builder = _build_scylladb(drop_old=False)

        assert db.name == "ScyllaDB"
        assert db.dim == 4
        # connect was called once
        builder.connect.assert_awaited_once()
        # keyspace creation + USE were called
        calls = [str(c) for c in session.execute.await_args_list]
        assert any("CREATE KEYSPACE" in c for c in calls)
        assert any("USE" in c for c in calls)
        # no DROP TABLE
        assert not any("DROP TABLE" in c for c in calls)

    def test_construction_drop_old_creates_table(self):
        """Constructor with drop_old=True must DROP + CREATE TABLE."""
        db, session, _builder = _build_scylladb(drop_old=True)

        calls = [str(c) for c in session.execute.await_args_list]
        assert any("DROP TABLE" in c for c in calls)
        assert any("CREATE TABLE" in c for c in calls)

    def test_construction_drop_old_creates_index_when_not_deferred(self):
        """When create_index_after_upload=False, index is created in __init__."""
        db, session, _builder = _build_scylladb(
            drop_old=True, create_index_after_upload=False,
        )
        calls = [str(c) for c in session.execute.await_args_list]
        assert any("CREATE CUSTOM INDEX" in c for c in calls)

    def test_construction_drop_old_no_index_when_deferred(self):
        """When create_index_after_upload=True (default), index is NOT created in __init__."""
        db, session, _builder = _build_scylladb(
            drop_old=True, create_index_after_upload=True,
        )
        calls = [str(c) for c in session.execute.await_args_list]
        assert not any("CREATE CUSTOM INDEX" in c for c in calls)


class TestScyllaDBInit:
    """Tests for the init() context-manager lifecycle."""

    def test_init_opens_and_closes_session(self):
        """init() should open a session and prepare INSERT; exit clears state."""
        db, session, _builder = _build_scylladb()

        # Before init(), session is None (constructor uses a throwaway session)
        assert db.session is None

        with (
            patch.object(
                type(db), "_build_session_builder",
                return_value=_mock_session_builder(session),
            ),
            patch.object(
                type(db), "_read_credentials",
                return_value=(None, None),
            ),
        ):
            with db.init():
                assert db.session is not None
                assert db.prepared_insert is not None

            # After exit, state is cleared
            assert db.session is None
            assert db.prepared_insert is None

    def test_ensure_session_raises_without_init(self):
        """_ensure_session must raise when called outside init()."""
        db, _, _ = _build_scylladb()
        with pytest.raises(RuntimeError, match="no active session"):
            db._ensure_session()


class TestScyllaDBInsertEmbeddings:
    """Tests for insert_embeddings() — verifies asyncio.gather pattern."""

    def _setup_db_with_session(self, with_scalar_labels=False):
        """Return (db, mock_session) with session + prepared insert wired up."""
        db, session, _builder = _build_scylladb(
            with_scalar_labels=with_scalar_labels,
        )
        prep = _mock_prepared()
        db.session = session
        db.prepared_insert = prep
        # Reset call counts accumulated during construction
        session.execute.reset_mock()
        session.prepare.reset_mock()
        return db, session, prep

    def test_insert_calls_execute_per_row(self):
        """Each embedding must produce one session.execute call via gather."""
        db, session, prep = self._setup_db_with_session()

        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        metadata = [1, 2]

        count, err = db.insert_embeddings(embeddings, metadata)

        assert count == 2
        assert err is None
        # Two execute calls (one per row), dispatched via asyncio.gather
        assert session.execute.await_count == 2

    def test_insert_with_labels(self):
        """With scalar labels enabled, label data must be passed to execute."""
        db, session, prep = self._setup_db_with_session(with_scalar_labels=True)

        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        metadata = [42]
        labels = ["red"]

        count, err = db.insert_embeddings(embeddings, metadata, labels_data=labels)

        assert count == 1
        assert err is None
        session.execute.assert_awaited_once()
        args = session.execute.await_args
        # values list should be [key, embedding, label]
        assert args[0][1] == [42, [0.1, 0.2, 0.3, 0.4], "red"]

    def test_insert_labels_required_when_scalar(self):
        """Omitting labels_data when with_scalar_labels=True must return error."""
        db, session, prep = self._setup_db_with_session(with_scalar_labels=True)

        count, err = db.insert_embeddings([[0.1, 0.2, 0.3, 0.4]], [1])

        assert count == 0
        assert isinstance(err, ValueError)

    def test_insert_returns_error_on_execute_failure(self):
        """If session.execute raises, error should be captured and returned."""
        db, session, prep = self._setup_db_with_session()
        session.execute = AsyncMock(side_effect=RuntimeError("connection lost"))

        count, err = db.insert_embeddings([[0.1, 0.2, 0.3, 0.4]], [1])

        assert count == 0
        assert isinstance(err, Exception)


class TestScyllaDBSearch:
    """Tests for prepare_filter() + search_embedding()."""

    def _setup_db_with_session(self):
        db, session, _builder = _build_scylladb()
        db.session = session
        return db, session

    def test_prepare_filter_non_filter(self):
        """NonFilter must prepare a query without WHERE clause."""
        from vectordb_bench.backend.filter import NonFilter

        db, session = self._setup_db_with_session()
        db.prepare_filter(NonFilter())

        session.prepare.assert_awaited_once()
        cql = session.prepare.await_args[0][0]
        assert "WHERE" not in cql
        assert "ORDER BY" in cql
        assert db._filter_params == ()

    def test_prepare_filter_num_ge(self):
        """NumGE filter must insert WHERE id > ? and ALLOW FILTERING."""
        from vectordb_bench.backend.filter import IntFilter

        db, session = self._setup_db_with_session()
        f = IntFilter(int_value=500, filter_rate=0.01)
        db.prepare_filter(f)

        cql = session.prepare.await_args[0][0]
        assert "WHERE id > ?" in cql
        assert "ALLOW FILTERING" in cql
        assert db._filter_params == (500,)

    def test_prepare_filter_str_equal(self):
        """StrEqual filter must insert WHERE label = ?."""
        from vectordb_bench.backend.filter import LabelFilter

        db, session = self._setup_db_with_session()
        f = LabelFilter(label_percentage=0.05)
        db.prepare_filter(f)

        cql = session.prepare.await_args[0][0]
        assert "WHERE filtering_label = ?" in cql
        assert db._filter_params == (f.label_value,)

    def test_search_embedding_returns_ids(self):
        """search_embedding must return list of IDs from result rows."""
        db, session = self._setup_db_with_session()

        # Mock result object with iter_rows returning dicts
        mock_result = MagicMock()
        mock_result.iter_rows.return_value = [
            {"id": 10}, {"id": 20}, {"id": 30},
        ]
        session.execute = AsyncMock(return_value=mock_result)

        # Set up prepared_lookup
        db.prepared_lookup = _mock_prepared()
        db._filter_params = ()

        ids = db.search_embedding([0.1, 0.2, 0.3, 0.4], k=3)

        assert ids == [10, 20, 30]
        session.execute.assert_awaited_once()

    def test_search_without_prepare_raises(self):
        """search_embedding must raise if prepare_filter was not called."""
        db, session = self._setup_db_with_session()

        with pytest.raises(RuntimeError, match="prepared_lookup is not set"):
            db.search_embedding([0.1, 0.2, 0.3, 0.4])

    def test_search_with_filter_params(self):
        """Filter params must be prepended to the execute values."""
        db, session = self._setup_db_with_session()

        mock_result = MagicMock()
        mock_result.iter_rows.return_value = [{"id": 5}]
        session.execute = AsyncMock(return_value=mock_result)

        db.prepared_lookup = _mock_prepared()
        db._filter_params = (100,)

        query = [0.1, 0.2, 0.3, 0.4]
        ids = db.search_embedding(query, k=10)

        args = session.execute.await_args
        values = args[0][1]
        # first element is filter param, then query vector, then k
        assert values == [100, query, 10]
        assert ids == [5]


class TestScyllaDBOptimize:
    """Tests for optimize() / _wait_for_index_build()."""

    def _setup_db_with_session(self):
        db, session, _builder = _build_scylladb()
        db.session = session
        return db, session

    def test_optimize_creates_index_when_deferred(self):
        """optimize() must CREATE INDEX when create_index_after_upload is True."""
        db, session = self._setup_db_with_session()
        assert db.case_config.create_index_after_upload is True

        # Make probe query succeed immediately
        session.execute = AsyncMock(return_value=MagicMock())

        db.optimize()

        calls = [str(c) for c in session.execute.await_args_list]
        assert any("CREATE CUSTOM INDEX" in c for c in calls)

    def test_optimize_skips_index_when_not_deferred(self):
        """optimize() must NOT CREATE INDEX when create_index_after_upload is False."""
        db, session, _builder = _build_scylladb(create_index_after_upload=False)
        db.session = session

        session.execute = AsyncMock(return_value=MagicMock())

        db.optimize()

        calls = [str(c) for c in session.execute.await_args_list]
        assert not any("CREATE CUSTOM INDEX" in c for c in calls)

    def test_wait_for_index_build_succeeds_immediately(self):
        """If probe query succeeds, _wait_for_index_build returns immediately."""
        db, session = self._setup_db_with_session()
        session.execute = AsyncMock(return_value=MagicMock())

        db._wait_for_index_build(timeout=5.0)

        # Probe was called at least once
        assert session.execute.await_count >= 1

    def test_wait_for_index_build_retries_then_succeeds(self):
        """Probe should retry on failure, then succeed."""
        db, session = self._setup_db_with_session()

        # Fail twice, then succeed
        session.execute = AsyncMock(
            side_effect=[RuntimeError("not ready"), RuntimeError("not ready"), MagicMock()]
        )

        with patch.object(_scylladb_module.time, "sleep"):
            db._wait_for_index_build(timeout=60.0, poll_interval=0.0)

        assert session.execute.await_count == 3

    def test_wait_for_index_build_timeout(self):
        """If index never becomes ready, TimeoutError is raised."""
        db, session = self._setup_db_with_session()
        session.execute = AsyncMock(side_effect=RuntimeError("not ready"))

        with patch.object(_scylladb_module.time, "sleep"):
            with pytest.raises(TimeoutError, match="index not ready"):
                db._wait_for_index_build(timeout=0.0, poll_interval=0.0)


class TestScyllaDBSchemaVariants:
    """Tests for table/index creation with different label/index scope configs."""

    def test_table_with_scalar_labels_global_index(self):
        """With scalar labels + GLOBAL scope, PK should be (id, label)."""
        db, session, _ = _build_scylladb(
            drop_old=True,
            with_scalar_labels=True,
            index_scope=ScyllaDBIndexScope.GLOBAL,
        )

        calls = [str(c) for c in session.execute.await_args_list]
        create_table = [c for c in calls if "CREATE TABLE" in c]
        assert len(create_table) == 1
        assert "PRIMARY KEY (id, filtering_label)" in create_table[0]

    def test_table_with_scalar_labels_local_index(self):
        """With scalar labels + LOCAL scope, PK should be (label, id)."""
        db, session, _ = _build_scylladb(
            drop_old=True,
            with_scalar_labels=True,
            index_scope=ScyllaDBIndexScope.LOCAL,
        )

        calls = [str(c) for c in session.execute.await_args_list]
        create_table = [c for c in calls if "CREATE TABLE" in c]
        assert len(create_table) == 1
        assert "PRIMARY KEY (filtering_label, id)" in create_table[0]

    def test_table_without_scalar_labels(self):
        """Without scalar labels, PK should be (id) only."""
        db, session, _ = _build_scylladb(
            drop_old=True,
            with_scalar_labels=False,
        )

        calls = [str(c) for c in session.execute.await_args_list]
        create_table = [c for c in calls if "CREATE TABLE" in c]
        assert len(create_table) == 1
        assert "PRIMARY KEY (id)" in create_table[0]

    def test_index_target_with_local_scope(self):
        """LOCAL scope index target should include partition key column."""
        db, session, _ = _build_scylladb(
            drop_old=True,
            with_scalar_labels=True,
            index_scope=ScyllaDBIndexScope.LOCAL,
            create_index_after_upload=False,
        )

        calls = [str(c) for c in session.execute.await_args_list]
        create_index = [c for c in calls if "CREATE CUSTOM INDEX" in c]
        assert len(create_index) == 1
        assert "((filtering_label), vector)" in create_index[0]

    def test_index_target_with_global_scope(self):
        """GLOBAL scope index target should be just (vector)."""
        db, session, _ = _build_scylladb(
            drop_old=True,
            with_scalar_labels=True,
            index_scope=ScyllaDBIndexScope.GLOBAL,
            create_index_after_upload=False,
        )

        calls = [str(c) for c in session.execute.await_args_list]
        create_index = [c for c in calls if "CREATE CUSTOM INDEX" in c]
        assert len(create_index) == 1
        assert "(vector)" in create_index[0]
        assert "((filtering_label)" not in create_index[0]


class TestScyllaDBNeedNormalize:
    """Test need_normalize_cosine behaviour."""

    def test_need_normalize_cosine_returns_true(self):
        db, _, _ = _build_scylladb()
        assert db.need_normalize_cosine() is True


class TestScyllaDBCredentials:
    """Test credential-reading logic."""

    def test_read_credentials_both_set(self):
        with patch("environs.Env") as MockEnv:
            env_instance = MagicMock()
            env_instance.read_env = MagicMock()
            env_instance.side_effect = lambda key, default=None: {
                "SCYLLADB_USERNAME": "admin",
                "SCYLLADB_PASSWORD": "secret",
            }.get(key, default)
            MockEnv.return_value = env_instance

            username, password = ScyllaDB._read_credentials()
            assert username == "admin"
            assert password == "secret"

    def test_read_credentials_none_set(self):
        with patch("environs.Env") as MockEnv:
            env_instance = MagicMock()
            env_instance.read_env = MagicMock()
            env_instance.side_effect = lambda key, default=None: default
            MockEnv.return_value = env_instance

            username, password = ScyllaDB._read_credentials()
            assert username is None
            assert password is None


class TestRunHelper:
    """Tests for the _run() async-to-sync bridge."""

    def test_run_executes_coroutine(self):
        async def coro():
            return 42

        assert _run(coro()) == 42

    def test_run_propagates_exception(self):
        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run(failing())
