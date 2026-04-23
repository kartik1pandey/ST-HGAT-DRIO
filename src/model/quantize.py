"""Quantization utilities for run_submission.py."""
from __future__ import annotations
import copy, time
from typing import Any, Dict
import torch
import torch.nn as nn


def benchmark_model(model: nn.Module, sample_batch: Dict[str, Any],
                    n_runs: int = 200, warmup: int = 30) -> Dict[str, float]:
    model.eval()
    lats = []
    with torch.inference_mode():
        for i in range(warmup + n_runs):
            t0 = time.perf_counter()
            _ = model(sample_batch)
            if i >= warmup:
                lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    return {
        "mean_ms": float(sum(lats) / len(lats)),
        "min_ms":  lats[0],
        "max_ms":  lats[-1],
        "p50_ms":  lats[len(lats) // 2],
        "p95_ms":  lats[int(len(lats) * 0.95)],
    }


def compare_outputs(fp32_model: nn.Module, int8_model: nn.Module,
                    sample_batch: Dict[str, Any]) -> Dict[str, float]:
    fp32_model.eval(); int8_model.eval()
    with torch.inference_mode():
        out_fp32 = fp32_model(sample_batch).float()
        out_int8 = int8_model(sample_batch).float()
    diff = (out_fp32 - out_int8).abs()
    return {
        "max_abs_error":  float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "rel_error":      float((diff / (out_fp32.abs() + 1e-8)).mean()),
    }


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization — works on all platforms."""
    m = copy.deepcopy(model).eval()
    try:
        return torch.quantization.quantize_dynamic(m, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    except Exception:
        return m


def prepare_qat(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Prepare model for QAT. Falls back to fbgemm on non-QNNPACK platforms."""
    m = copy.deepcopy(model).train()
    try:
        torch.backends.quantized.engine = backend
        m.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
        torch.ao.quantization.prepare_qat(m, inplace=True)
    except Exception:
        torch.backends.quantized.engine = "fbgemm"
        m.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
        torch.ao.quantization.prepare_qat(m, inplace=True)
    return m


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert QAT model to INT8."""
    m = copy.deepcopy(model).eval()
    torch.ao.quantization.convert(m, inplace=True)
    return m
