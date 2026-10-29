"""Track a box through a numbered JPEG frame directory."""

import argparse

from sam2_inference import VideoTracker

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("frames")
parser.add_argument("--box", type=float, nargs=4, required=True)
args = parser.parse_args()

tracker = VideoTracker(args.checkpoint)
tracker.open(args.frames)
tracker.add_prompt(0, 1, box=args.box)
for result in tracker.track():
    print(result.frame_index, int(result.masks[0].sum()))
tracker.close()
