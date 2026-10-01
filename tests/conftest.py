"""
Shared pytest setup.

`pytest.ini` puts `src/` on sys.path, so the tests import the same top-level modules the
scripts use (constraints, sample, stp_preflight, check_constraints, ml_models, ...).
"""

import os

# Must be set before matplotlib is imported anywhere (ml_models.evaluate imports pyplot):
# tests must never need a display or Qt.
os.environ.setdefault("MPLBACKEND", "Agg")
