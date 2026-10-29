"""GPU 사용 가능 여부에 따라 PyTorch와 패키지를 설치한다."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", choices=["12.6", "12.8", "13.0"], default="12.6",
                        help="GPU 설치 시 사용할 CUDA wheel 버전 (기본: 12.6)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
        available = probe.returncode == 0 and bool(probe.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        available = False
    gpu = args.device == "cuda" or (args.device == "auto" and available)
    wheel = "cu" + args.cuda.replace(".", "") if gpu else "cpu"
    root = str(Path(__file__).resolve().parents[1])
    extra = "cuda" + args.cuda.split(".")[0] if gpu else ""
    target = root + (f"[{extra}]" if extra else "")
    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch",
         "--index-url", f"https://download.pytorch.org/whl/{wheel}"],
        [sys.executable, "-m", "pip", "install", "-e", target],
    ]
    print(f"Selected: {wheel}. CUDA driver compatibility must match --cuda.", flush=True)
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    if gpu and not args.dry_run:
        subprocess.run([sys.executable, "-c",
                        "import torch, cupy as cp; "
                        "assert torch.cuda.is_available(), 'CUDA is not available'; "
                        "x = torch.ones((2, 2), device='cuda'); "
                        "cp.linalg.svd(cp.from_dlpack(x)); print('CUDA/CuPy OK')"], check=True)


if __name__ == "__main__":
    main()
