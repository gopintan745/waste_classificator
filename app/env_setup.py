"""
Environment configuration. Must be imported BEFORE any other modules.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress OpenMP duplicate library warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Reduce noisy warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
