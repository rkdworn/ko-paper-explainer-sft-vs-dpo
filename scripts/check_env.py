"""학습 환경 점검: 라이브러리 버전과 GPU 인식 여부를 확인한다.

실행: uv run python scripts/check_env.py
"""

import importlib.metadata as md
import sys

import torch


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows에서 한글 출력이 깨지지 않도록
    print(f"Python        {sys.version.split()[0]}")
    for pkg in ["torch", "transformers", "trl", "peft", "datasets", "accelerate"]:
        print(f"{pkg:<13} {md.version(pkg)}")

    print()
    if not torch.cuda.is_available():
        print("CUDA 사용 불가: GPU를 찾지 못했습니다. PyTorch CUDA 빌드가 설치됐는지 확인하세요.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"GPU           {props.name}")
    print(f"VRAM          {props.total_memory / 1024**3:.1f} GB")
    print(f"Compute cap.  {props.major}.{props.minor}")
    print(f"CUDA (torch)  {torch.version.cuda}")
    print(f"bf16 지원     {torch.cuda.is_bf16_supported()}")

    # 실제로 GPU에서 연산이 도는지 확인
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    print(f"GPU 연산 테스트 통과 (결과값 {y:.1f})")


if __name__ == "__main__":
    main()
