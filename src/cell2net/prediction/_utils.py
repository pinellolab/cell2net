import logging
import os
import random
import warnings
from typing import Literal

import numpy as np
import torch
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)
from scvi import settings
from scvi.utils._docstrings import devices_dsp

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def random_seq(seq_len):
    bases = ["A", "C", "G", "T"]
    rand_seq = "".join([np.random.choice(bases) for i in range(seq_len)])
    return rand_seq


@devices_dsp.dedent
def parse_device_args(
    accelerator: str = "auto",
    devices: int | list[int] | str = "auto",
    return_device: Literal["torch", "jax"] | None = None,
    validate_single_device: bool = False,
):
    """Parses device-related arguments.

    Parameters
    ----------
    %(param_accelerator)s
    %(param_devices)s
    %(param_return_device)s
    %(param_validate_single_device)s
    """
    valid = [None, "torch", "jax"]
    if return_device not in valid:
        raise ValueError(f"`return_device` must be one of {valid}")

    _validate_single_device = validate_single_device and devices != "auto"
    cond1 = isinstance(devices, list) and len(devices) > 1
    cond2 = isinstance(devices, str) and "," in devices
    cond3 = devices == -1
    if _validate_single_device and (cond1 or cond2 or cond3):
        raise ValueError("Only a single device can be specified for `device`.")

    connector = _AcceleratorConnector(accelerator=accelerator, devices=devices)
    _accelerator = connector._accelerator_flag
    _devices = connector._devices_flag

    if _accelerator in ["tpu", "ipu", "hpu"]:
        warnings.warn(
            f"The selected accelerator `{_accelerator}` has not been extensively "
            "tested in scvi-tools. Please report any issues in the GitHub repo.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )
    elif _accelerator == "mps" and accelerator == "auto":
        # auto accelerator should not default to mps
        connector = _AcceleratorConnector(accelerator="cpu", devices=devices)
        _accelerator = connector._accelerator_flag
        _devices = connector._devices_flag
    elif _accelerator == "mps" and accelerator != "auto":
        warnings.warn(
            "`accelerator` has been set to `mps`. Please note that not all PyTorch "
            "operations are supported with this backend. Refer to "
            "https://github.com/pytorch/pytorch/issues/77764 for more details.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )

    # get the first device index
    if isinstance(_devices, list):
        device_idx = _devices[0]
    elif isinstance(_devices, str) and "," in _devices:
        device_idx = _devices.split(",")[0]
    else:
        device_idx = _devices

    if devices == "auto" and _accelerator != "cpu":
        # auto device should not use multiple devices for non-cpu accelerators
        _devices = [device_idx]

    device = torch.device("cpu")
    if _accelerator != "cpu":
        device = torch.device(f"{_accelerator}:{device_idx}")
    return _accelerator, _devices, device
