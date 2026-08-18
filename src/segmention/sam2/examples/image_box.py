"""Segment an RGB image with one xyxy box."""

import argparse

import numpy as np
from PIL import Image

from sam2_inference import ImageSegmenter

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("image")
parser.add_argument("--box", type=float, nargs=4, required=True)
args = parser.parse_args()

rgb = np.asarray(Image.open(args.image).convert("RGB"))
result = ImageSegmenter(args.checkpoint).segment(
    rgb, box=args.box, multimask=False
)
print(result.masks.shape, result.scores)
