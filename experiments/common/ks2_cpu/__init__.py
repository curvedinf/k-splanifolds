"""Frozen rank-8 KS2 CPU exports: FP32 Horner and GS128 INT8 VNNI."""

from .inference import KS2Horner, KS2VNNI

__all__ = ['KS2Horner', 'KS2VNNI']
