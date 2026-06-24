import torch
from typing import Optional, Union


def is_xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    is_avail = getattr(xpu, "is_available", None)
    if is_avail is None:
        return False
    try:
        return bool(is_avail())
    except Exception:
        return False


def resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if isinstance(device, torch.device):
        dev = device
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        if dev.type == "xpu" and not is_xpu_available():
            raise RuntimeError("device=xpu requested but torch.xpu.is_available() is False")
        if dev.type == "cuda" and getattr(torch.cuda, "set_device", None) is not None and dev.index is not None:
            torch.cuda.set_device(int(dev.index))
        if dev.type == "xpu":
            xpu = getattr(torch, "xpu", None)
            set_dev = getattr(xpu, "set_device", None) if xpu is not None else None
            if set_dev is not None and dev.index is not None:
                set_dev(int(dev.index))
        return dev

    device_str = str(device or "auto").strip().lower()
    if device_str in ("auto", "best"):
        if is_xpu_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
    if dev.type == "xpu" and not is_xpu_available():
        raise RuntimeError("device=xpu requested but torch.xpu.is_available() is False")

    if dev.type == "cuda" and getattr(torch.cuda, "set_device", None) is not None and dev.index is not None:
        torch.cuda.set_device(int(dev.index))
    if dev.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        set_dev = getattr(xpu, "set_device", None) if xpu is not None else None
        if set_dev is not None and dev.index is not None:
            set_dev(int(dev.index))

    return dev


def manual_seed_all(device: torch.device, seed: int) -> None:
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        return
    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        fn = getattr(xpu, "manual_seed_all", None) if xpu is not None else None
        if fn is not None:
            fn(seed)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def try_load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: Optional[dict],
    device: torch.device,
) -> bool:
    if not isinstance(state_dict, dict):
        return False
    try:
        optimizer.load_state_dict(state_dict)
    except (ValueError, KeyError, RuntimeError):
        return False
    move_optimizer_state_to_device(optimizer, device)
    return True
