#!/usr/bin/env python3
"""Standard CLI wrapper for the DRIFT-SENSE Procedural SEM Dataset Generator."""

import sys
from pathlib import Path

# Add dataset directory to path
dataset_dir = Path(__file__).parent / "dataset"
sys.path.insert(0, str(dataset_dir))

from dataset_generator import main

if __name__ == "__main__":
    main()
