"""EXIF reading that explicitly ignores GPS location tags."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from PIL import ExifTags, Image

from astrogeo.pipeline.result_schema import ExifInfo


def _ratio_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(Fraction(value[0], value[1]))
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value.numerator) / float(value.denominator)
        return float(value)
    except Exception:
        return None


def read_exif(image_path: str) -> ExifInfo:
    path = Path(image_path)
    info = ExifInfo()
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return info
            tag_names = {key: ExifTags.TAGS.get(key, str(key)) for key in exif.keys()}
            raw = {tag_names[key]: exif.get(key) for key in exif.keys()}
            info.raw_tags = {
                key: str(value)
                for key, value in raw.items()
                if not key.startswith("GPS")
            }
            info.gps_present_ignored = any(key.startswith("GPS") for key in raw)
            info.capture_time_raw = raw.get("DateTimeOriginal") or raw.get("DateTime")
            info.offset_time_raw = raw.get("OffsetTimeOriginal") or raw.get("OffsetTime")
            info.focal_length_mm = _ratio_to_float(raw.get("FocalLength"))
            info.focal_length_35mm = _ratio_to_float(raw.get("FocalLengthIn35mmFilm"))
            info.lens_model = raw.get("LensModel")
            info.camera_model = raw.get("Model")
            info.orientation = raw.get("Orientation")
            return info
    except Exception:
        return info

