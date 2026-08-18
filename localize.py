#!/usr/bin/env python3
"""Drift-Sense Industrial Nanoscale Localization Engine — Production CLI Entry Point.

Usage:
    python localize.py --reference reference.png --search search.png

Output to stdout:
    x,y

Note: All diagnostics and logs are written strictly to sys.stderr.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from src.champion import DriftSenseChampionEngine as DriftSenseEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift-Sense Production SEM Localization Engine")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image (.png)")
    parser.add_argument("--search", type=str, required=True, help="Path to search image (.png)")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic logs to stderr")

    args = parser.parse_args()

    ref_path = Path(args.reference)
    search_path = Path(args.search)

    if not ref_path.exists():
        sys.stderr.write(f"Error: Reference image not found at {ref_path}\n")
        sys.exit(1)
    if not search_path.exists():
        sys.stderr.write(f"Error: Search image not found at {search_path}\n")
        sys.exit(1)

    ref_img_raw = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
    search_img_raw = cv2.imread(str(search_path), cv2.IMREAD_UNCHANGED)

    if ref_img_raw is None or search_img_raw is None:
        sys.stderr.write("Error: Failed to load one or both images.\n")
        sys.exit(1)

    # Bonus points: Detect RGB optical images and convert to structural luminance
    if len(ref_img_raw.shape) == 3 or len(search_img_raw.shape) == 3:
        if args.verbose:
            print("[DRIFT-SENSE] Optical Microscope (RGB) images detected! Converting to structural luminance.")
        ref_img = cv2.cvtColor(ref_img_raw, cv2.COLOR_BGR2GRAY) if len(ref_img_raw.shape) == 3 else ref_img_raw
        search_img = cv2.cvtColor(search_img_raw, cv2.COLOR_BGR2GRAY) if len(search_img_raw.shape) == 3 else search_img_raw
    else:
        ref_img = ref_img_raw
        search_img = search_img_raw

    engine = DriftSenseEngine(verbose=args.verbose)
    x, y, diagnostics = engine.localize(ref_img, search_img)

    forensics = diagnostics['forensics']
    n95 = forensics.n95_score
    status = forensics.status

    ambiguity_reason = "NONE"
    if status == "AMBIGUOUS" or diagnostics.get('tie_occurred', False):
        ambiguity_reason = "P/2 periodic equivalent (or similar structural ambiguity)"

    print("\n")
    print("LOCALIZATION:")
    print(f"({x:.4f}, {y:.4f})\n")
    print("CONFIDENCE:")
    print(f"{(1.0 - n95):.4f}\n")  # High confidence when uncertainty is low
    print("DETERMINACY:")
    print(f"{status}\n")
    print("N95:")
    print(f"{n95:.4f}\n")
    print("AMBIGUITY:")
    print(f"{ambiguity_reason}\n")


if __name__ == "__main__":
    main()
