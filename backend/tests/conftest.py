"""Test environment: an isolated data directory and database, created before the package is imported."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="indicator_lab_test_")
os.environ["IL_DATA_DIR"] = _TMP
os.environ["IL_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["IL_OPTUNA_STORAGE"] = f"sqlite:///{os.path.join(_TMP, 'optuna.db')}"
os.environ["IL_INDICATOR_PLUGIN_DIR"] = os.path.join(_TMP, "plugins")
os.environ.setdefault("IL_MIN_TRADES", "5")

import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root():
    return _TMP


@pytest.fixture(scope="session")
def dataset_id(tmp_root):
    from indicatorlab.data import generate_synthetic, import_csv
    from indicatorlab.db import get_db
    from indicatorlab.registry import REGISTRY

    get_db()
    REGISTRY.discover(plugin_dir=os.path.join(tmp_root, "plugins"))
    path = generate_synthetic(os.path.join(tmp_root, "synth.csv"), days=25, seed=3)
    return import_csv(path, name="test_synth", symbol="SYNTHETIC", replace=True)["id"]
