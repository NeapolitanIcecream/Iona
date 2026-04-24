import numpy as np
from PIL import Image

from iona.solver import image_variants


def test_variant_generation_keeps_original_when_masked_variant_fails(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), color=(0, 0, 0)).save(image_path)

    def raise_masked(path, mask):  # noqa: ARG001
        raise OSError("cannot write masked image")

    monkeypatch.setattr(image_variants, "_make_masked_variant", raise_masked)
    monkeypatch.setattr(image_variants, "_make_star_enhanced_variant", lambda path, mask: str(tmp_path / "stars.png"))

    variants = image_variants.make_solver_image_variants(str(image_path), np.ones((6, 8), dtype=bool))

    assert [(variant.label, variant.path, variant.temporary) for variant in variants] == [
        ("original", str(image_path), False),
        ("star_enhanced", str(tmp_path / "stars.png"), True),
    ]


def test_variant_generation_returns_created_variants_when_later_variant_fails(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), color=(0, 0, 0)).save(image_path)
    masked_path = tmp_path / "masked.png"
    masked_path.write_bytes(b"temporary")

    def raise_enhanced(path, mask):  # noqa: ARG001
        raise OSError("cannot write enhanced image")

    monkeypatch.setattr(image_variants, "_make_masked_variant", lambda path, mask: str(masked_path))
    monkeypatch.setattr(image_variants, "_make_star_enhanced_variant", raise_enhanced)

    variants = image_variants.make_solver_image_variants(str(image_path), np.ones((6, 8), dtype=bool))

    assert [(variant.label, variant.path, variant.temporary) for variant in variants] == [
        ("original", str(image_path), False),
        ("sky_masked", str(masked_path), True),
    ]
    image_variants.cleanup_solver_image_variants(variants)
    assert not masked_path.exists()


def test_temp_variant_file_is_removed_when_save_fails(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), color=(0, 0, 0)).save(image_path)
    temp_path = tmp_path / "variant.png"

    class FakeTempFile:
        name = str(temp_path)

        def close(self) -> None:
            temp_path.write_bytes(b"created")

    class UnsavableImage:
        def save(self, path) -> None:  # noqa: ARG002
            raise OSError("save failed")

    monkeypatch.setattr(image_variants, "NamedTemporaryFile", lambda **_kwargs: FakeTempFile())
    monkeypatch.setattr(image_variants.Image, "fromarray", lambda array: UnsavableImage())

    try:
        image_variants._make_masked_variant(str(image_path), np.ones((6, 8), dtype=bool))
    except OSError:
        pass

    assert not temp_path.exists()
