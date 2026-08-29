from PIL import Image

from input.image_loader import load_image_frames


def test_alpha_is_composited_onto_white_background(tmp_path):
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    for x in range(5):
        for y in range(10):
            img.putpixel((x, y), (10, 20, 30, 255))
    path = tmp_path / "alpha.png"
    img.save(path)

    frames = load_image_frames(path)
    assert len(frames) == 1
    out = frames[0].image
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (10, 20, 30)  # opaque half keeps its color
    assert out.getpixel((9, 0)) == (255, 255, 255)  # transparent half -> white


def test_exif_orientation_is_applied(tmp_path):
    img = Image.new("RGB", (20, 10), (255, 0, 0))
    # 왼쪽 절반만 파란색으로 칠해 방향 보정 여부를 픽셀로 확인할 수 있게 한다.
    for x in range(10):
        for y in range(10):
            img.putpixel((x, y), (0, 0, 255))

    exif = img.getexif()
    exif[274] = 6  # Orientation: 6 = 시계 방향 90도 회전 필요
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif.tobytes())

    frames = load_image_frames(path)
    out = frames[0].image
    # 회전 적용 결과 가로/세로가 뒤바뀐다.
    assert out.size == (10, 20)


def test_multi_frame_tiff_preserves_order(tmp_path):
    frame1 = Image.new("RGB", (5, 5), (255, 0, 0))
    frame2 = Image.new("RGB", (5, 5), (0, 255, 0))
    path = tmp_path / "multi.tiff"
    frame1.save(path, save_all=True, append_images=[frame2])

    frames = load_image_frames(path)
    assert [f.index for f in frames] == [1, 2]
    assert frames[0].image.getpixel((0, 0)) == (255, 0, 0)
    assert frames[1].image.getpixel((0, 0)) == (0, 255, 0)
