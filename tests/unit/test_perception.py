"""Unit tests for perception API."""
import pytest
from sam3_perception.models.sam3_wrapper import (
    ConceptQuery,
    VisualQuery,
    SegmentationResult,
    SAM3Model,
)


class TestConceptQuery:
    def test_default_values(self):
        query = ConceptQuery(text="test")
        assert query.text == "test"
        assert query.confidence_threshold == 0.5
        assert query.max_instances == 100

    def test_custom_threshold(self):
        query = ConceptQuery(text="forklift", confidence_threshold=0.8)
        assert query.confidence_threshold == 0.8


class TestVisualQuery:
    def test_point_query(self):
        query = VisualQuery(
            points=[(100, 100), (200, 200)],
            point_labels=[1, 0]
        )
        assert len(query.points) == 2
        assert query.point_labels == [1, 0]

    def test_box_query(self):
        query = VisualQuery(boxes=[(10, 10, 100, 100)])
        assert len(query.boxes) == 1


class TestSegmentationResult:
    def test_to_dict(self):
        result = SegmentationResult(
            instances=[],
            image_size=(1920, 1080),
            processing_time_ms=45.2
        )
        d = result.to_dict()
        assert d["image_size"] == (1920, 1080)
        assert d["instance_count"] == 0


@pytest.mark.asyncio
async def test_model_load():
    """Test model loading."""
    model = SAM3Model(
        model_path="/tmp/mock",
        device="cpu",
        inference_provider="local",
    )
    await model.load()
    assert model.is_loaded
