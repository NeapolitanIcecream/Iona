import numpy as np

from iona.cv import segmentation


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeProcessor:
    def __call__(self, images, return_tensors):  # noqa: ARG002
        return {}

    def post_process_semantic_segmentation(self, outputs, target_sizes):  # noqa: ARG002
        labels = np.zeros((8, 10), dtype=np.int64)
        labels[:4, :] = 2
        labels[4:, :6] = 1
        labels[4:, 6:] = 4
        return [_FakeTensor(labels)]


class _FakeModel:
    class Config:
        id2label = {1: "building, edifice", 2: "sky", 4: "tree"}

    config = Config()

    def eval(self):
        return None

    def __call__(self, **inputs):  # noqa: ARG002
        return object()


def test_segformer_scene_masks_map_sky_and_building_labels(monkeypatch) -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(
        segmentation,
        "_load_segformer_model",
        lambda model_id: (_FakeTorch(), _FakeProcessor(), _FakeModel()),
    )

    result = segmentation.estimate_scene_masks(
        image,
        backend="segformer",
        model_id="fake/segformer",
    )

    assert result.backend == "segformer"
    assert result.model_id == "fake/segformer"
    assert result.sky.sky_mask[:4, :].all()
    assert not result.sky.sky_mask[4:, :].any()
    assert result.building_mask[4:, :6].all()
    assert not result.building_mask[:, 6:].any()
    assert result.diagnostics["sky_label_ids"] == [2]
    assert result.diagnostics["building_label_ids"] == [1]


def test_auto_segmentation_falls_back_to_classic_with_diagnostics(monkeypatch) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:20, :] = 8
    image[20:, :] = 80

    def raise_missing_dependency(model_id):  # noqa: ARG001
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(segmentation, "_load_segformer_model", raise_missing_dependency)

    result = segmentation.estimate_scene_masks(image, backend="auto", model_id="fake/segformer")

    assert result.backend == "classic"
    assert result.used_fallback
    assert result.fallback_reason == "segformer_unavailable"
    assert result.diagnostics["requested_backend"] == "auto"
    assert "SegFormer segmentation unavailable; using classic CV masks." in result.warnings
