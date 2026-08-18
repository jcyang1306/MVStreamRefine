import os
import subprocess
import sys
from pathlib import Path


def test_src_layout_imports_outside_repository(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    code = """
from importlib.resources import files
import sam2
import sam2_inference
assert 'packages/sam2_inference/src/sam2' in sam2.__file__.replace('\\\\', '/')
assert files('sam2').joinpath('configs/sam2.1/sam2.1_hiera_t.yaml').is_file()
assert sam2_inference.ImageSegmenter.__module__ == 'sam2_inference.image'
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
