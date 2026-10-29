"""Minimal stream example; replace frame_source() with a camera adapter."""

import argparse

import numpy as np
from PIL import Image

from sam2_inference import StreamTracker

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("frames", nargs="+")
parser.add_argument("--box", type=float, nargs=4, required=True)
args = parser.parse_args()

frames = [np.asarray(Image.open(path).convert("RGB")) for path in args.frames]
tracker = StreamTracker(args.checkpoint)
tracker.add_prompt(frames[0], box=args.box)
for frame in frames[1:]:
    result = tracker.track(frame)
    print(result.frame_index, int(result.masks[0].sum()))
tracker.close()
