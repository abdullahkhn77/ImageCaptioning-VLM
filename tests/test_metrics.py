"""Tests for evaluation metrics."""

import pytest

from vlm_caption_finetune_mlops.evaluation.metrics import (
    compute_bleu,
    compute_meteor,
    compute_rouge,
    compute_metrics,
)


def test_compute_bleu():
    refs = [["a cat on the mat"], ["a dog in the sun"]]
    hyps = ["a cat on the mat", "a dog in the sun"]
    out = compute_bleu(refs, hyps)
    assert "bleu" in out
    assert 0 <= out["bleu"] <= 100


def test_compute_rouge():
    refs = ["a cat on the mat", "a dog in the sun"]
    hyps = ["a cat on the mat", "a dog in the sun"]
    out = compute_rouge(refs, hyps)
    assert "rouge1" in out and "rougeL" in out
    assert out["rougeL"] > 0


def test_compute_metrics():
    refs = ["a cat on the mat", "a dog in the sun"]
    hyps = ["a cat on the mat", "a dog in the sun"]
    out = compute_metrics(refs, hyps)
    assert "bleu" in out or "meteor" in out or "rouge1" in out
