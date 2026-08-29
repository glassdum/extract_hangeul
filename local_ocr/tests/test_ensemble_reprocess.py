import numpy as np
from PIL import Image

from ensemble.reprocess import VariantCandidate, pick_best, reprocess_crop
from preprocess.variants import generate_variants
from recognition.base import OCREngine, RecognizedItem


class StubEngine(OCREngine):
    def __init__(self, item):
        self._item = item

    def recognize(self, image):
        return [self._item] if self._item is not None else []


def _blank():
    return Image.new("RGB", (10, 10), "white")


def test_pick_best_selects_highest_confidence():
    candidates = [
        VariantCandidate(variant="original", image=_blank(), item=RecognizedItem("a", 0.4, [])),
        VariantCandidate(
            variant="binarize_otsu", image=_blank(), item=RecognizedItem("b", 0.7, [])
        ),
        VariantCandidate(
            variant="upscale_2x", image=_blank(), item=RecognizedItem("c", 0.55, [])
        ),
    ]
    best = pick_best(candidates)
    assert best.variant == "binarize_otsu"
    assert best.item.text == "b"


def test_pick_best_returns_none_when_nothing_detected():
    candidates = [VariantCandidate(variant="original", image=_blank(), item=None)]
    assert pick_best(candidates) is None


def test_reprocess_crop_covers_original_plus_every_generated_variant():
    img = Image.new("RGB", (40, 20), "white")
    item = RecognizedItem(text="stub", confidence=0.5, polygon=[[0, 0], [10, 0], [10, 5], [0, 5]])

    candidates = reprocess_crop(img, StubEngine(item))

    expected_variants = {"original", *generate_variants(img).keys()}
    assert {c.variant for c in candidates} == expected_variants
    assert all(c.item is not None and c.item.text == "stub" for c in candidates)


def test_reprocess_crop_when_engine_finds_nothing():
    img = Image.new("RGB", (40, 20), "white")
    candidates = reprocess_crop(img, StubEngine(None))
    assert all(c.item is None for c in candidates)
    assert pick_best(candidates) is None


def test_reprocess_crop_keeps_the_image_each_variant_actually_used():
    img = Image.new("RGB", (40, 20), "white")
    item = RecognizedItem(text="stub", confidence=0.5, polygon=[[0, 0], [10, 0], [10, 5], [0, 5]])

    candidates = reprocess_crop(img, StubEngine(item))

    original = next(c for c in candidates if c.variant == "original")
    assert original.image is img  # 원본은 그대로

    variants = generate_variants(img)
    other = next(c for c in candidates if c.variant != "original")
    assert np.array_equal(np.array(other.image), np.array(variants[other.variant]))
