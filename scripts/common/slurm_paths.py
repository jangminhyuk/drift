"""
Project-wide path constants for the GNSS/IMU port. Centralized so that runners,
test scripts, and SLURM job templates all agree on filesystem locations.
"""
import os

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

# Raw and preprocessed dataset on a large data allocation
DATA_RAW_ROOT     = os.path.join(os.environ.get('WRAP_DATA_ROOT','data'), 'i2nav_robot/raw')
DATA_PREPROC_ROOT = os.path.join(os.environ.get('WRAP_DATA_ROOT','data'), 'i2nav_robot/preprocessed')
CACHE_GNSS_ROOT   = os.path.join(os.environ.get('WRAP_DATA_ROOT','data'), 'i2nav_robot/cache_gnss')

# Results and checkpoints on /projects (longer retention)
RESULTS_GNSS_ROOT = os.path.join(os.environ.get('WRAP_GNSS_ROOT','gnss_out'), 'results')
CKPT_GNSS_ROOT    = os.path.join(os.environ.get('WRAP_GNSS_ROOT','gnss_out'), 'ckpts')

# External reference repos (git clones under the repo)
KFGINS_REPO  = os.path.join(REPO_ROOT, 'external', 'KF-GINS')
KFGINS_BUILD = os.path.join(KFGINS_REPO, 'bin')
I2NAV_REPO   = os.path.join(REPO_ROOT, 'external', 'i2Nav-Robot')

# C++ DR solver build dir
DREKF_CPP_BUILD = os.path.join(REPO_ROOT, 'scripts', 'drekf_cpp', 'build')

# Conda environment
CONDA_ENV = os.path.join(os.environ.get('WRAP_STORE','store'), 'conda_envs/dreskf')
