"""Print where live separation will run: gpu, cpu, cpu-build, cuda-unavailable or missing.

Used by the launchers; exit code is always 0 so startup can explain instead of failing.
"""
import importlib.util

if importlib.util.find_spec("torch") is None:
    print("missing")
else:
    import torch
    if torch.cuda.is_available():
        print(f"gpu {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    elif torch.version.cuda is None:
        print(f"cpu-build (torch {torch.__version__})")
    else:
        print(f"cuda-unavailable (torch {torch.__version__}, CUDA {torch.version.cuda})")
