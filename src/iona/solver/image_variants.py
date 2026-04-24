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
    masked = _make_masked_variant(image_path, sky_mask)
    enhanced = _make_star_enhanced_variant(image_path, sky_mask)
    variants.extend(
        [
            SolverImageVariant("sky_masked", masked, True),
            SolverImageVariant("star_enhanced", enhanced, True),
        ]
    )
    return variants


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
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).copy()
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.shape == array.shape[:2]:
        array[~mask] = 0
    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(array).save(tmp.name)
    return tmp.name


def _make_star_enhanced_variant(image_path: str, sky_mask: np.ndarray) -> str:
    image = Image.open(image_path).convert("L")
    gray = np.asarray(image).astype(float)
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.shape == gray.shape and np.any(mask):
        values = gray[mask]
        low, high = np.quantile(values, [0.60, 0.995])
        enhanced = np.clip((gray - low) / max(high - low, 1.0), 0, 1) * 255.0
        enhanced[~mask] = 0
    else:
        enhanced = gray
    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8)).save(tmp.name)
    return tmp.name
