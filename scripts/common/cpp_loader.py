"""
Helper that makes `dr_ekf_cpp` (the pybind11 binding of the DR-MMSE-TAC/CDC
solver in scripts/drekf_cpp/) importable from anywhere in the repo without
requiring PYTHONPATH setup.

Usage:
    from cpp_loader import import_dr_ekf_cpp
    dr_ekf_cpp = import_dr_ekf_cpp()
    result = dr_ekf_cpp.solve_dr_mmse_tac_factored(...)
"""
import os
import sys

_BUILD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'drekf_cpp', 'build')
)


def import_dr_ekf_cpp():
    """Ensure the dr_ekf_cpp .so is on sys.path, then import and return it."""
    if _BUILD_DIR not in sys.path:
        sys.path.insert(0, _BUILD_DIR)
    import dr_ekf_cpp  # noqa: E402
    return dr_ekf_cpp


def get_build_dir():
    return _BUILD_DIR
