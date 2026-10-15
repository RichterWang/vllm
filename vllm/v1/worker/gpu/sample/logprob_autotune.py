# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional one-time calibration of the token-logprob split count."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.gpu.sample import logprob

_BATCHES = (1, 8, 16, 32, 64, 128)
_SPLITS = (1, 2, 4, 8, 16, 32, 64)
_VERSION = 1


def _original_pr_split(batch: int, vocab: int, sms: int) -> int:
    count = min(2 * sms // batch, triton.cdiv(vocab, 8192), 64)
    return 1 if count < 2 else 1 << (count.bit_length() - 1)


def _fingerprint(vocab: int, dtype: torch.dtype, k: int, device: torch.device) -> dict:
    device_index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    capability = current_platform.get_device_capability(device_index)
    return {
        "version": _VERSION,
        "gpu_uuid": current_platform.get_device_uuid(device_index),
        "gpu_name": current_platform.get_device_name(device_index),
        "gpu_capability": (
            [capability.major, capability.minor] if capability is not None else None
        ),
        "gpu_sms": num_compute_units(device_index),
        "gpu_memory": current_platform.get_device_total_memory(device_index),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "kernel_sha256": hashlib.sha256(
            Path(logprob.__file__).read_bytes()
        ).hexdigest(),
        "calibrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vocab_size": vocab,
        "dtype": str(dtype).removeprefix("torch."),
        "num_logprobs": k,
        "batches": list(_BATCHES),
        "splits": list(_SPLITS),
    }


def _measure(call, repetitions: int = 12) -> float:
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        call()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) * 1e6 / repetitions


def _calibrate(
    vocab: int, dtype: torch.dtype, k: int, device: torch.device
) -> list[dict]:
    rows = []
    device_index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    sms = num_compute_units(device_index)
    for batch in _BATCHES:
        # Avoid large temporary allocations at startup on very large vocabularies.
        if batch * vocab * torch.empty((), dtype=dtype).element_size() > 128 * 1024**2:
            continue
        logits = torch.randn((batch, vocab), dtype=dtype, device=device)
        ids = torch.randint(vocab, (batch, k), device=device)
        reference = torch.log_softmax(logits.double(), dim=-1).gather(-1, ids).float()
        timings: dict[int, list[float]] = {}
        for split in _SPLITS:
            call = (
                lambda s=split, x=logits, token_ids=ids: logprob.compute_token_logprobs(
                    x, token_ids, s
                )
            )
            try:
                result = call()
                torch.accelerator.synchronize()
                torch.testing.assert_close(result, reference, atol=2e-3, rtol=2e-3)
                _measure(call, 3)
                timings[split] = []
            except (RuntimeError, AssertionError) as error:
                if isinstance(error, torch.OutOfMemoryError):
                    torch.accelerator.empty_cache()
                continue
        for order in (list(timings), list(reversed(timings)), list(timings)):
            for split in order:
                call = (
                    lambda s=split,
                    x=logits,
                    token_ids=ids: logprob.compute_token_logprobs(x, token_ids, s)
                )
                timings[split].append(_measure(call))
        medians = {s: sorted(ts)[len(ts) // 2] for s, ts in timings.items()}
        original = _original_pr_split(batch, vocab, sms)
        if original not in medians:
            raise RuntimeError(f"PR split {original} failed calibration at B={batch}")
        fastest = min(medians, key=lambda split: medians[split])
        chosen = fastest if medians[fastest] < medians[original] * 0.95 else original
        rows.append(
            {
                "batch_size": batch,
                "chosen_splits": chosen,
                "pr_splits": original,
                "wall_us": medians,
            }
        )
        del logits, ids, reference
    if not rows:
        raise RuntimeError("No token-logprob calibration shapes fit in memory")
    return rows


def initialize_split_autotune(
    vocab_size: int,
    dtype: torch.dtype,
    num_logprobs: int = 6,
    device: torch.device | str = "cuda",
    cache_dir: Path | None = None,
) -> dict:
    """Load a hardware/source-specific table or calibrate it once under a file lock."""
    started = time.perf_counter()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Token-logprob calibration requires CUDA")
    fingerprint = _fingerprint(vocab_size, dtype, num_logprobs, device)
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    root = Path(
        cache_dir
        or os.getenv("VLLM_TOKEN_LOGPROB_AUTOTUNE_CACHE_DIR")
        or Path(envs.VLLM_CACHE_ROOT) / "token_logprob_autotune"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.json"
    with (root / f"{key}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            saved = json.loads(path.read_text()) if path.exists() else None
            if saved is not None and (
                saved.get("fingerprint") != fingerprint
                or not saved.get("rows")
                or [r.get("batch_size") for r in saved["rows"]]
                != sorted(set(r.get("batch_size") for r in saved["rows"]))
                or any(
                    r.get("batch_size") not in _BATCHES
                    or r.get("chosen_splits") not in _SPLITS
                    for r in saved["rows"]
                )
            ):
                saved = None
        except (OSError, ValueError, TypeError, AttributeError):
            saved = None
        if saved is None:
            try:
                rows = _calibrate(vocab_size, dtype, num_logprobs, device)
                torch.accelerator.synchronize()
                calibration_seconds = time.perf_counter() - started
                saved = {
                    "fingerprint": fingerprint,
                    "rows": rows,
                    "calibration_seconds": calibration_seconds,
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=root, delete=False
                ) as tmp:
                    json.dump(saved, tmp, indent=2)
                    temp_path = tmp.name
                os.replace(temp_path, path)
                status = "calibrated"
            finally:
                torch.accelerator.empty_cache()
        else:
            status = "cache_hit"
    logprob._set_autotuned_split_profile(
        {
            "vocab_size": vocab_size,
            "dtype": fingerprint["dtype"],
            "num_logprobs": num_logprobs,
            "rows": saved["rows"],
        },
        device,
    )
    return {
        "status": status,
        "init_seconds": time.perf_counter() - started,
        "calibration_seconds": saved["calibration_seconds"],
        "cache_path": str(path),
        "fingerprint": fingerprint,
        "rows": saved["rows"],
    }
