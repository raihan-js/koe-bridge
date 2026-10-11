"""Smoke tests (no models — interface only)."""
import numpy as np


def test_wer_math():
    from jiwer import wer

    assert wer(["hello world"], ["hello world"]) == 0.0
    assert wer(["a b c"], ["a x c"]) > 0
