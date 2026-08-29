import numpy as np
from PIL import Image

from common.types import BBox
from preprocess.binarize import binarize_adaptive, binarize_otsu
from preprocess.contrast import adjust_gamma, apply_clahe, stretch_contrast
from preprocess.crop import crop_with_padding
from preprocess.denoise import denoise_bilateral, denoise_median
from preprocess.geometry import deskew, estimate_skew_angle
from preprocess.invert import invert
from preprocess.lines import remove_table_lines
from preprocess.quality import is_likely_inverted, is_low_contrast
from preprocess.upscale import upscale
from preprocess.variants import generate_variants


def _checkerboard(size=64):
    array = np.zeros((size, size), dtype=np.uint8)
    array[: size // 2, : size // 2] = 255
    array[size // 2 :, size // 2 :] = 255
    return Image.fromarray(array).convert("RGB")


def test_upscale_doubles_dimensions():
    img = Image.new("RGB", (40, 20), "white")
    out = upscale(img, factor=2)
    assert out.size == (80, 40)


def test_binarize_otsu_produces_two_tone_image():
    img = _checkerboard()
    out = binarize_otsu(img)
    colors = {out.getpixel((x, y)) for x in range(0, out.width, 4) for y in range(0, out.height, 4)}
    assert colors <= {(0, 0, 0), (255, 255, 255)}


def test_binarize_adaptive_preserves_size():
    img = _checkerboard()
    out = binarize_adaptive(img)
    assert out.size == img.size


def test_invert_reverses_pixels():
    img = Image.new("RGB", (10, 10), (10, 20, 30))
    out = invert(img)
    assert out.getpixel((0, 0)) == (245, 235, 225)


def test_is_low_contrast_detects_flat_image():
    flat = np.full((30, 30), 128, dtype=np.uint8)
    assert is_low_contrast(flat) is True

    checker = np.array(_checkerboard().convert("L"))
    assert is_low_contrast(checker) is False


def test_is_likely_inverted_detects_dark_border():
    dark_border = np.zeros((20, 20), dtype=np.uint8)
    dark_border[5:15, 5:15] = 255
    assert is_likely_inverted(dark_border) is True

    light_border = np.full((20, 20), 255, dtype=np.uint8)
    light_border[5:15, 5:15] = 0
    assert is_likely_inverted(light_border) is False


def test_apply_clahe_and_stretch_and_gamma_preserve_size():
    img = _checkerboard()
    assert apply_clahe(img).size == img.size
    assert stretch_contrast(img).size == img.size
    assert adjust_gamma(img).size == img.size


def test_denoise_preserves_size():
    img = _checkerboard()
    assert denoise_median(img).size == img.size
    assert denoise_bilateral(img).size == img.size


def test_estimate_skew_angle_near_zero_for_axis_aligned_block():
    array = np.full((100, 200), 255, dtype=np.uint8)
    array[40:60, 20:180] = 0  # 수평 막대: 기울기 없음
    angle = estimate_skew_angle(array)
    assert abs(angle) < 2.0


def test_estimate_skew_angle_and_deskew_round_trip():
    array = np.full((200, 400), 255, dtype=np.uint8)
    array[90:110, 40:360] = 0  # 수평 막대(텍스트 줄 대용)
    img = Image.fromarray(array).convert("RGB")

    rotated = img.rotate(8, expand=False, fillcolor=(255, 255, 255))
    angle = estimate_skew_angle(np.array(rotated.convert("L")))
    assert abs(angle - (-8)) < 1.0

    fixed = deskew(rotated, angle)
    angle_after = estimate_skew_angle(np.array(fixed.convert("L")))
    assert abs(angle_after) < 1.0


def test_deskew_preserves_size():
    img = Image.new("RGB", (100, 60), "white")
    out = deskew(img, angle=5.0)
    assert out.size == img.size


def test_remove_table_lines_clears_long_horizontal_line():
    array = np.full((100, 100), 255, dtype=np.uint8)
    array[50, :] = 0  # 전체 폭을 가로지르는 표 선
    img = Image.fromarray(array).convert("RGB")

    out = remove_table_lines(img)
    out_array = np.array(out.convert("L"))
    assert out_array[50, :].min() > 200  # 선이 흰색으로 지워졌다


def test_crop_with_padding_clips_to_image_bounds():
    img = Image.new("RGB", (50, 50), "white")
    bbox = BBox(0, 0, 10, 10)
    out = crop_with_padding(img, bbox, pad_ratio=1.0)
    assert out.size[0] <= 50 and out.size[1] <= 50


def test_generate_variants_includes_contrast_fix_only_when_low_contrast():
    low_contrast_img = Image.fromarray(np.full((40, 40), 128, dtype=np.uint8)).convert("RGB")
    variants = generate_variants(low_contrast_img)
    assert "clahe" in variants
    assert "upscale_2x" in variants  # 항상 시도하는 후보

    high_contrast_img = _checkerboard()
    variants2 = generate_variants(high_contrast_img)
    assert "clahe" not in variants2
