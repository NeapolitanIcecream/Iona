"""Image variants shared by plate solver backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SolverImageVariant:
    label: str
    path: str
    temporary: bool = False


def make_solver_image_variants(image_path: str, sky_mask: Optional[np.ndarray]) -> List[SolverImageVariant]:
    variants = [SolverImageVariant("original", image_path, False)]
    if sky_mask is None:
        return variants
    _append_temp_variant(variants, "sky_masked", _make_masked_variant, image_path, sky_mask)
    _append_temp_variant(variants, "star_enhanced", _make_star_enhanced_variant, image_path, sky_mask)
    return variants


def _append_temp_variant(
    variants: List[SolverImageVariant],
    label: str,
    maker,
    image_path: str,
    sky_mask: np.ndarray,
) -> None:
    try:
        variants.append(SolverImageVariant(label, maker(image_path, sky_mask), True))
    except Exception:
        return


def cleanup_solver_image_variants(variants: List[SolverImageVariant]) -> None:
    for variant in variants:
        if not variant.temporary:
            continue
        try:
            Path(variant.path).unlink(missing_ok=True)
        except TypeError:
            try:
                Path(variant.path).unlink()
            except FileNotFoundError:
                pass


def _make_masked_variant(image_path: str, sky_mask: np.ndarray) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    array = np.asarray(image).copy()
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.shape == array.shape[:2]:
        array[~mask] = 0
    return _save_temp_image(Image.fromarray(array))


def _make_star_enhanced_variant(image_path: str, sky_mask: np.ndarray) -> str:
    with Image.open(image_path) as image:
        image = image.convert("L")
    gray = np.asarray(image).astype(float)
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.shape == gray.shape and np.any(mask):
        values = gray[mask]
        low, high = np.quantile(values, [0.60, 0.995])
        enhanced = np.clip((gray - low) / max(high - low, 1.0), 0, 1) * 255.0
        enhanced[~mask] = 0
    else:
        enhanced = gray
    return _save_temp_image(Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8)))


def _save_temp_image(image: Image.Image) -> str:
    tmp_path = None
    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        image.save(tmp_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return tmp_path
