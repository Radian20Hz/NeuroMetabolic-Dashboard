"""Explicit, versioned CPU strict / CUDA seeded FP32 process settings."""
import os
import torch

REVISION = "nmd-numerics-2"


def profile(device):
    if device not in ("cpu", "cuda"):
        raise ValueError("Unsupported numerical device")
    return dict(revision=REVISION, name="cpu-strict" if device == "cpu" else "cuda-seeded",
                deterministic_algorithms=True, warn_only=device == "cuda",
                cudnn_benchmark=False, cudnn_deterministic=True,
                fp32_precision="ieee", cuda_matmul_precision="ieee", cudnn_precision="ieee",
                matmul_precision="highest", cublas_workspace=":4096:8", precision="32-true",
                amp=False, num_workers=0)


def effective():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                fp32_precision=torch.backends.fp32_precision,
                cuda_matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                cudnn_precision=torch.backends.cudnn.fp32_precision,
                matmul_precision=torch.get_float32_matmul_precision(),
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                precision="32-true", amp=torch.is_autocast_enabled("cuda"), num_workers=0)


def configure(device):
    wanted=profile(device)
    # New PyTorch precision API only; do not mix with legacy allow_tf32 setters.
    torch.backends.fp32_precision="ieee"
    torch.backends.cuda.matmul.fp32_precision="ieee"
    torch.backends.cudnn.fp32_precision="ieee"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=wanted["cublas_workspace"]
    torch.use_deterministic_algorithms(True, warn_only=wanted["warn_only"])
    verify(device)
    return wanted


def verify(device):
    actual=effective()
    wanted=profile(device)
    for key,value in actual.items():
        if value != wanted[key]:
            raise ValueError(f"Effective numerical profile mismatch: {key}")
    return actual
