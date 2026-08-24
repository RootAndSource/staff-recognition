"""StaffRecognition - 服务器 GPU torch 诊断与安装脚本（跨平台）。

用法：
    python scripts/tools/install_gpu_torch.py                 # 只诊断，不改动
    python scripts/tools/install_gpu_torch.py --install       # 诊断 + 重装 GPU 版 torch
    python scripts/tools/install_gpu_torch.py --cuda 121      # 手动指定 CUDA 版本（可选）

说明：
- 用 `python -m pip` 而不是裸 `pip`，避免装进错误的 Python 环境；
- Windows 下默认 PyPI 的 torch 是 CPU 版，必须走 PyTorch 官方 index-url；
- A6000（Ampere）建议 cu121 / cu124，按 nvidia-smi 驱动自动推荐。
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """运行命令并返回结果（不抛异常）。"""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    return subprocess.run(cmd, **kwargs)


def detect_torch() -> dict:
    """读取当前解释器里 torch 的信息（不存在则返回空）。"""
    code = (
        "import torch; "
        "print(torch.__version__); "
        "print(getattr(torch.version, 'cuda', None) or ''); "
        "print(torch.cuda.is_available())"
    )
    r = _run([sys.executable, "-c", code])
    if r.returncode != 0:
        return {"installed": False}
    parts = r.stdout.strip().splitlines()
    return {
        "installed": True,
        "version": parts[0] if parts else "",
        "cuda": parts[1] if len(parts) > 1 else "",
        "available": (parts[2] if len(parts) > 2 else "False") == "True",
    }


def detect_driver() -> dict:
    """从 nvidia-smi 读取驱动版本与驱动支持的 CUDA 版本。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"found": False}
    r = _run([exe])
    if r.returncode != 0:
        return {"found": False}
    text = r.stdout
    driver = ""
    cuda = ""
    m = re.search(r"Driver Version:\s*([\d.]+)", text)
    if m:
        driver = m.group(1)
    m = re.search(r"CUDA Version:\s*([\d.]+)", text)
    if m:
        cuda = m.group(1)
    return {"found": True, "driver": driver, "cuda_supported": cuda}


def recommend_cuda(driver_cuda: str, manual: int | None) -> int:
    """按驱动支持的 CUDA 版本推荐 wheel 的 cu 标签；可手动覆盖。"""
    if manual is not None:
        return manual
    try:
        major, minor = (int(x) for x in driver_cuda.split(".")[:2])
    except (ValueError, AttributeError):
        return 121
    if (major, minor) >= (12, 8):
        return 128
    if (major, minor) >= (12, 4):
        return 124
    if (major, minor) >= (12, 1):
        return 121
    if (major, minor) >= (11, 8):
        return 118
    return 121  # 驱动过旧时仍尝试 cu121，让用户自行判断


def uninstall_torch() -> bool:
    print("\n[1/3] 卸载当前 torch / torchvision ...")
    r = _run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"])
    if r.returncode != 0:
        print(r.stderr.strip() or r.stdout.strip())
        return False
    return True


def install_torch(cuda: int) -> bool:
    print(f"\n[2/3] 安装 GPU 版 torch（cu{cuda}，PyTorch 官方源）...")
    url = f"https://download.pytorch.org/whl/cu{cuda}"
    r = _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            url,
        ]
    )
    if r.returncode != 0:
        print(r.stderr.strip() or r.stdout.strip())
        return False
    return True


def verify() -> dict:
    print("\n[3/3] 验证 ...")
    info = detect_torch()
    if not info.get("installed"):
        print("torch 未安装")
        return info
    print(f"torch    : {info['version']}")
    print(f"cuda tag : {info['cuda'] or '(无, CPU 版)'}")
    print(f"cuda ok  : {info['available']}")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="服务器 GPU torch 诊断与安装")
    parser.add_argument("--install", action="store_true", help="诊断后重装 GPU 版 torch")
    parser.add_argument("--cuda", type=int, default=None, help="手动指定 CUDA 版本，如 121")
    args = parser.parse_args()

    print("=" * 60)
    print("StaffRecognition - torch / CUDA 环境诊断")
    print("=" * 60)
    print(f"系统        : {platform.platform()}")
    print(f"Python      : {sys.executable} ({platform.python_version()})")

    torch_info = detect_torch()
    if torch_info.get("installed"):
        print(f"torch      : {torch_info['version']}")
        print(f"cuda tag   : {torch_info['cuda'] or '(无, CPU 版)'}")
        print(f"cuda 可用  : {torch_info['available']}")
    else:
        print("torch      : 未安装")

    driver = detect_driver()
    if driver.get("found"):
        print(f"GPU 驱动   : {driver['driver']}（支持 CUDA {driver['cuda_supported']}）")
    else:
        print("GPU 驱动   : 未检测到 nvidia-smi（显卡驱动未装或不在 PATH）")

    if not args.install:
        print("\n提示：运行 --install 可自动重装 GPU 版 torch。")
        return 0

    if not driver.get("found"):
        print("\n错误：未检测到 NVIDIA 驱动，请先安装显卡驱动。", file=sys.stderr)
        return 1

    cuda = recommend_cuda(driver["cuda_supported"], args.cuda)
    print(f"\n将安装 cu{cuda} 版 torch（驱动支持 CUDA {driver['cuda_supported']}）")
    if not uninstall_torch():
        print("卸载失败，中止。", file=sys.stderr)
        return 1
    if not install_torch(cuda):
        print("安装失败，请检查网络或换 cu 版本重试。", file=sys.stderr)
        return 1
    info = verify()
    if info.get("available"):
        print("\n✅ GPU 版 torch 就绪。")
        return 0
    print("\n⚠️  torch 已装但 CUDA 不可用，请检查驱动与 cu 版本匹配。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
