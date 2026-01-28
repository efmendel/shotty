"""Tests for services/models.py - SwingAnalysisResults data model."""

import json
import pytest
from services.models import SwingAnalysisResults


class TestInitialization:
    """Tests for SwingAnalysisResults initialization."""

    def test_empty_phases(self):
        results = SwingAnalysisResults()
        assert all(v is None for v in results.phases.values())
        assert len(results.phases) == 5

    def test_empty_engine(self):
        results = SwingAnalysisResults()
        assert all(v is None for v in results.engine.values())

    def test_empty_tempo(self):
        results = SwingAnalysisResults()
        assert all(v is None for v in results.tempo.values())

    def test_empty_kinetic_chain(self):
        results = SwingAnalysisResults()
        assert all(v is None for v in results.kinetic_chain.values())

    def test_empty_quality(self):
        results = SwingAnalysisResults()
        assert results.video_quality is None
        assert results.tracking_quality is None


class TestAddPhase:
    """Tests for add_phase method."""

    def test_add_detected_phase(self):
        results = SwingAnalysisResults()
        results.add_phase("contact", detected=True, frame=102, timestamp=3.4, confidence=0.95)
        phase = results.phases["contact"]
        assert phase["detected"] is True
        assert phase["frame"] == 102
        assert phase["timestamp"] == 3.4
        assert phase["confidence"] == 0.95

    def test_add_undetected_phase(self):
        results = SwingAnalysisResults()
        results.add_phase("unit_turn", detected=False, confidence=0.0)
        phase = results.phases["unit_turn"]
        assert phase["detected"] is False

    def test_add_phase_with_metrics(self):
        results = SwingAnalysisResults()
        results.add_phase(
            "contact", detected=True, frame=100, confidence=0.9,
            wrist_velocity=0.82, elbow_angle=165.3
        )
        phase = results.phases["contact"]
        assert phase["wrist_velocity"] == 0.82
        assert phase["elbow_angle"] == 165.3

    def test_invalid_phase_name(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="Invalid phase_name"):
            results.add_phase("invalid_phase", detected=True, confidence=0.9)

    def test_invalid_detected_type(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="detected must be a boolean"):
            results.add_phase("contact", detected="yes", confidence=0.9)

    def test_confidence_too_high(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="confidence must be between"):
            results.add_phase("contact", detected=True, confidence=1.5)

    def test_confidence_too_low(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="confidence must be between"):
            results.add_phase("contact", detected=True, confidence=-0.1)


class TestAddEngineMetrics:
    """Tests for add_engine_metrics method."""

    def test_add_all_metrics(self):
        results = SwingAnalysisResults()
        results.add_engine_metrics(
            hip_shoulder_sep={"max_value": 35.2, "frame": 67},
            max_shoulder_rot={"value": -42.1, "frame": 67},
            max_hip_rot={"value": -55.3, "frame": 65},
        )
        assert results.engine["hip_shoulder_separation"]["max_value"] == 35.2
        assert results.engine["max_shoulder_rotation"]["value"] == -42.1
        assert results.engine["max_hip_rotation"]["value"] == -55.3

    def test_invalid_type(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="must be a dict"):
            results.add_engine_metrics(hip_shoulder_sep="invalid")


class TestAddTempoMetrics:
    """Tests for add_tempo_metrics method."""

    def test_add_all_metrics(self):
        results = SwingAnalysisResults()
        results.add_tempo_metrics(
            backswing_duration=1.2,
            forward_swing_duration=0.3,
            swing_rhythm_ratio=4.0,
        )
        assert results.tempo["backswing_duration"] == 1.2
        assert results.tempo["forward_swing_duration"] == 0.3
        assert results.tempo["swing_rhythm_ratio"] == 4.0

    def test_negative_duration(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="non-negative"):
            results.add_tempo_metrics(backswing_duration=-1.0)


class TestAddKineticChainMetrics:
    """Tests for add_kinetic_chain_metrics method."""

    def test_add_all_metrics(self):
        results = SwingAnalysisResults()
        sequence = {"hip": {"frame": 65, "velocity": 245.3}}
        chain_lag = {"hip_to_shoulder": 0.06}
        results.add_kinetic_chain_metrics(
            sequence=sequence, chain_lag=chain_lag, confidence=0.92
        )
        assert results.kinetic_chain["peak_velocity_sequence"] == sequence
        assert results.kinetic_chain["chain_lag"] == chain_lag
        assert results.kinetic_chain["confidence"] == 0.92

    def test_invalid_confidence(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="confidence must be between"):
            results.add_kinetic_chain_metrics(confidence=1.5)


class TestQualityMetrics:
    """Tests for set_video_quality and set_tracking_quality."""

    def test_set_video_quality(self):
        results = SwingAnalysisResults()
        quality = {"resolution": {"width": 1920, "height": 1080}}
        results.set_video_quality(quality)
        assert results.video_quality == quality

    def test_set_tracking_quality(self):
        results = SwingAnalysisResults()
        tracking = {"detection_rate": 0.95}
        results.set_tracking_quality(tracking)
        assert results.tracking_quality == tracking

    def test_invalid_video_quality_type(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="must be a dict"):
            results.set_video_quality("invalid")

    def test_invalid_tracking_quality_type(self):
        results = SwingAnalysisResults()
        with pytest.raises(ValueError, match="must be a dict"):
            results.set_tracking_quality("invalid")


class TestSerialization:
    """Tests for to_dict and to_json."""

    def test_to_dict_keys(self):
        results = SwingAnalysisResults()
        d = results.to_dict()
        assert set(d.keys()) == {
            "phases", "engine", "tempo", "kinetic_chain",
            "video_quality", "tracking_quality",
        }

    def test_to_dict_roundtrip(self):
        results = SwingAnalysisResults()
        results.add_phase("contact", detected=True, frame=100, confidence=0.9)
        d = results.to_dict()
        assert d["phases"]["contact"]["detected"] is True

    def test_to_json_valid(self):
        results = SwingAnalysisResults()
        results.add_phase("contact", detected=True, frame=100, confidence=0.9)
        json_str = results.to_json()
        parsed = json.loads(json_str)
        assert parsed["phases"]["contact"]["frame"] == 100


class TestUtilityMethods:
    """Tests for get_phases_detected_count, get_overall_confidence, __repr__."""

    def test_phases_detected_count_zero(self):
        results = SwingAnalysisResults()
        assert results.get_phases_detected_count() == 0

    def test_phases_detected_count(self):
        results = SwingAnalysisResults()
        results.add_phase("contact", detected=True, frame=100, confidence=0.9)
        results.add_phase("backswing", detected=True, frame=50, confidence=0.8)
        results.add_phase("unit_turn", detected=False, confidence=0.0)
        assert results.get_phases_detected_count() == 2

    def test_overall_confidence_zero(self):
        results = SwingAnalysisResults()
        assert results.get_overall_confidence() == 0.0

    def test_overall_confidence(self):
        results = SwingAnalysisResults()
        results.add_phase("contact", detected=True, frame=100, confidence=0.9)
        results.add_phase("backswing", detected=True, frame=50, confidence=0.7)
        assert abs(results.get_overall_confidence() - 0.8) < 0.01

    def test_repr(self):
        results = SwingAnalysisResults()
        results.add_phase("contact", detected=True, frame=100, confidence=0.9)
        repr_str = repr(results)
        assert "phases_detected=1/5" in repr_str
        assert "overall_confidence=0.90" in repr_str