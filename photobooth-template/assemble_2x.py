#!/usr/bin/env python3
"""Build the 2x (3840x5720) template from an AI-upscaled frame.

The colour comes from a super-resolved copy of the frame (Real-ESRGAN x4,
reduced to 2x). The alpha is NOT taken from the upscaler: the slot rectangles
are re-drawn exactly at 2x, and the decorations' alpha inside each slot is
resampled from the 1x template, so slot edges stay perfectly straight.

Usage: assemble_2x.py template_transparent.png frame_2x.png OUT_DIR
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from extract_template import COL_X, ROW_Y, SLOT_W, SLOT_H, GREEN


def main(template_1x, frame_2x, out_dir):
    out_dir = Path(out_dir)
    a1 = cv2.imread(str(template_1x), cv2.IMREAD_UNCHANGED)[..., 3]
    frame = cv2.imread(str(frame_2x))
    H, W = a1.shape
    assert frame.shape[:2] == (2 * H, 2 * W), "frame must be exactly 2x the template"

    alpha = np.full((2 * H, 2 * W), 255, np.uint8)
    for y in ROW_Y:
        for x in COL_X:
            slot = a1[y:y + SLOT_H, x:x + SLOT_W]
            up = cv2.resize(slot, (2 * SLOT_W, 2 * SLOT_H), interpolation=cv2.INTER_CUBIC)
            up[up < 6] = 0          # no faint haze left over the cleared photo area
            up[up > 249] = 255
            alpha[2 * y:2 * (y + SLOT_H), 2 * x:2 * (x + SLOT_W)] = up

    frame[alpha == 0] = 0
    cv2.imwrite(str(out_dir / "template_transparent_2x.png"), np.dstack([frame, alpha]))
    af = (alpha.astype(np.float32) / 255)[..., None]
    green = np.empty_like(frame)
    green[:] = GREEN
    cv2.imwrite(str(out_dir / "template_green_2x.png"),
                np.round(frame * af + green * (1 - af)).astype(np.uint8))
    print("2x size:", frame.shape[1], "x", frame.shape[0],
          "| cleared px:", int((alpha == 0).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
