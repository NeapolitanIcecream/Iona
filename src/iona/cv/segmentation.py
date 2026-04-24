"""Scene segmentation backends for sky and foreground masks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

from iona.pipeline.result_schema import SceneMaskResult, SkyMaskResult, bounded

from .sky_mask import estimate_sky_mask


DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGMENTATION_BACKENDS = {"auto", "classic", "segformer"}
SKY_LABELS = {"sky"}
BUILDING_LABELS = {
    "building",
    "wall",
    "house",
    "skyscraper",
    "tower",
    "railing",
    "fence",
    "column",
    "banister",
    "bannister",
    "windowpane",
    "door",
}


def estimate_scene_masks(
    image: np.ndarray,
    backend: str = "auto",
    model_id: str = DEFAULT_SEGFORMER_MODEL,
) -> SceneMaskResult:
    """Estimate sky and building masks, preferring SegFormer when requested."""

    if backend not in SEGMENTATION_BACKENDS:
        raise ValueError(f"Unsupported segmentation backend: {backend}")
    if backend == "classic":
        return _classic_scene_masks(image, requested_backend=backend)

    try:
        result = _estimate_segformer_scene_masks(image, model_id=model_id, requested_backend=backend)
        if _implausible_mask_reason(result):
            reason = _implausible_mask_reason(result) or "implausible_segmentation"
            return _classic_scene_masks(
                image,
                requested_backend=backend,
                used_fallback=True,
                fallback_reason=reason,
                fallback_warning="SegFormer segmentation looked implausible; using classic CV masks.",
                model_id=model_id,
            )
        return result
    except Exception as exc:
        return _classic_scene_masks(
            image,
            requested_backend=backend,
            used_fallback=True,
            fallback_reason="segformer_unavailable",
            fallback_warning="SegFormer segmentation unavailable; using classic CV masks.",
            model_id=model_id,
            fallback_error=str(exc),
        )


def _classic_scene_masks(
    image: np.ndarray,
    requested_backend: str,
    used_fallback: bool = False,
    fallback_reason: Optional[str] = None,
    fallback_warning: Optional[str] = None,
    model_id: Optional[str] = None,
    fallback_error: Optional[str] = None,
) -> SceneMaskResult:
    sky = estimate_sky_mask(image)
    building_mask = None if sky.sky_mask is None else ~np.asarray(sky.sky_mask, dtype=bool)
    building_fraction = _mask_fraction(building_mask)
    warnings = list(sky.warnings)
    if fallback_warning:
        warnings.append(fallback_warning)
    diagnostics: Dict[str, Any] = {
        "requested_backend": requested_backend,
        "backend": "classic",
        "method": "classic_sky_inverse",
        "sky_fraction": sky.sky_fraction,
        "building_fraction": building_fraction,
    }
    if fallback_reason:
        diagnostics["fallback_reason"] = fallback_reason
    if fallback_error:
        diagnostics["fallback_error"] = fallback_error
    confidence = min(0.70, sky.confidence) if used_fallback else sky.confidence
    return SceneMaskResult(
        sky=sky,
        building_mask=building_mask,
        backend="classic",
        model_id=model_id if used_fallback else None,
        confidence=float(confidence),
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _estimate_segformer_scene_masks(
    image: np.ndarray,
    model_id: str,
    requested_backend: str,
) -> SceneMaskResult:
    torch, processor, model = _load_segformer_model(model_id)
    rgb = np.asarray(image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    pil_image = Image.fromarray(rgb).convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    segmentation = _segmentation_to_array(
        processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
    )
    id2label = _normalized_id2label(getattr(getattr(model, "config", None), "id2label", {}))
    sky_label_ids = _matching_label_ids(id2label, SKY_LABELS)
    building_label_ids = _matching_label_ids(id2label, BUILDING_LABELS)
    if not sky_label_ids:
        raise RuntimeError("SegFormer model did not expose a sky label.")
    if not building_label_ids:
        raise RuntimeError("SegFormer model did not expose building-like labels.")

    sky_mask = np.isin(segmentation, sky_label_ids)
    building_mask = np.isin(segmentation, building_label_ids)
    sky_mask, building_mask = _clean_segmentation_masks(sky_mask, building_mask)
    sky_fraction = _mask_fraction(sky_mask)
    building_fraction = _mask_fraction(building_mask)
    confidence = _segmentation_confidence(sky_fraction, building_fraction)
    warnings = []
    if confidence < 0.45:
        warnings.append("SegFormer segmentation mask areas are weak for this image.")

    sky = SkyMaskResult(
        sky_mask=sky_mask,
        confidence=float(confidence),
        sky_fraction=sky_fraction,
        warnings=list(warnings),
        diagnostics={
            "method": "segformer_semantic_segmentation",
            "model_id": model_id,
            "sky_label_ids": sky_label_ids,
            "building_label_ids": building_label_ids,
            "building_fraction": building_fraction,
        },
    )
    diagnostics = {
        "requested_backend": requested_backend,
        "backend": "segformer",
        "model_id": model_id,
        "sky_fraction": sky_fraction,
        "building_fraction": building_fraction,
        "sky_label_ids": sky_label_ids,
        "building_label_ids": building_label_ids,
    }
    return SceneMaskResult(
        sky=sky,
        building_mask=building_mask,
        backend="segformer",
        model_id=model_id,
        confidence=float(confidence),
        used_fallback=False,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _load_segformer_model(model_id: str) -> Tuple[Any, Any, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
    return torch, processor, model


def _segmentation_to_array(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.int64)


def _normalized_id2label(id2label: Any) -> Dict[int, str]:
    normalized: Dict[int, str] = {}
    if not isinstance(id2label, dict):
        return normalized
    for raw_id, raw_label in id2label.items():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        normalized[label_id] = _normalize_label(str(raw_label))
    return normalized


def _matching_label_ids(id2label: Dict[int, str], wanted: Iterable[str]) -> list[int]:
    wanted_normalized = {_normalize_label(label) for label in wanted}
    return sorted(
        label_id
        for label_id, label in id2label.items()
        if any(term in wanted_normalized for term in _label_terms(label))
    )


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", " ").replace("_", " ")


def _label_terms(label: str) -> list[str]:
    normalized = _normalize_label(label)
    terms = [normalized]
    for separator in (",", ";", "/"):
        if separator in normalized:
            terms.extend(part.strip() for part in normalized.split(separator))
    return [term for term in terms if term]


def _clean_segmentation_masks(sky_mask: np.ndarray, building_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if min(sky_mask.shape[:2]) < 32:
        return sky_mask.astype(bool), building_mask.astype(bool)
    try:
        import cv2
    except Exception:
        return sky_mask.astype(bool), building_mask.astype(bool)

    sky_u8 = sky_mask.astype(np.uint8)
    building_u8 = building_mask.astype(np.uint8)
    sky_kernel = np.ones((5, 5), np.uint8)
    building_kernel = np.ones((3, 3), np.uint8)
    sky_clean = cv2.morphologyEx(sky_u8, cv2.MORPH_CLOSE, sky_kernel)
    sky_clean = cv2.morphologyEx(sky_clean, cv2.MORPH_OPEN, sky_kernel)
    building_clean = cv2.morphologyEx(building_u8, cv2.MORPH_CLOSE, building_kernel)
    building_clean = cv2.dilate(building_clean, building_kernel, iterations=1)
    return sky_clean.astype(bool), building_clean.astype(bool) & ~sky_clean.astype(bool)


def _mask_fraction(mask: Optional[np.ndarray]) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.mean(np.asarray(mask, dtype=bool)))


def _segmentation_confidence(sky_fraction: float, building_fraction: float) -> float:
    sky_score = 1.0 - min(abs(sky_fraction - 0.35) / 0.35, 1.0)
    building_score = min(building_fraction / 0.12, 1.0)
    return bounded(0.25 + 0.50 * sky_score + 0.25 * building_score)


def _implausible_mask_reason(result: SceneMaskResult) -> Optional[str]:
    sky_fraction = result.sky.sky_fraction
    building_fraction = _mask_fraction(result.building_mask)
    if sky_fraction < 0.03 or sky_fraction > 0.92:
        return "implausible_sky_fraction"
    if building_fraction < 0.005:
        return "implausible_building_fraction"
    return None
