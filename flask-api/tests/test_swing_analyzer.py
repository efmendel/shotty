"""Tests for services/swing_analyzer.py - biomechanical analysis algorithms."""

import pytest
from services.swing_analyzer import (
    SwingAnalyzerConfig,
    SwingAnalyzer,
    PRESET_STANDARD,
    PRESET_SENSITIVE,
    PRESET_STRICT,
)


class TestSwingAnalyzerConfig:
    """Tests for SwingAnalyzerConfig."""

    def test_default_config(self):
        config = SwingAnalyzerConfig()
        assert config.velocity_threshold == 0.5
        assert config.contact_angle_min == 150
        assert config.use_adaptive_velocity is False
        assert config.adaptive_velocity_percent == 0.15
        assert config.contact_frame_offset == 3
        assert config.follow_through_offset == 0.15
        assert config.forward_swing_search_window == 40
        assert config.min_valid_frames == 10
        assert config.kinematic_chain_mode is False
        assert config.contact_detection_method == "velocity_peak"

    def test_custom_config(self):
        config = SwingAnalyzerConfig(
            velocity_threshold=0.7,
            contact_angle_min=160,
            kinematic_chain_mode=True,
            contact_detection_method="hybrid",
        )
        assert config.velocity_threshold == 0.7
        assert config.contact_angle_min == 160
        assert config.kinematic_chain_mode is True
        assert config.contact_detection_method == "hybrid"

    def test_invalid_velocity_threshold(self):
        with pytest.raises(ValueError, match="velocity_threshold"):
            SwingAnalyzerConfig(velocity_threshold=-1)

    def test_invalid_contact_angle_min_low(self):
        with pytest.raises(ValueError, match="contact_angle_min"):
            SwingAnalyzerConfig(contact_angle_min=-1)

    def test_invalid_contact_angle_min_high(self):
        with pytest.raises(ValueError, match="contact_angle_min"):
            SwingAnalyzerConfig(contact_angle_min=181)

    def test_invalid_adaptive_velocity_percent(self):
        with pytest.raises(ValueError, match="adaptive_velocity_percent"):
            SwingAnalyzerConfig(adaptive_velocity_percent=0.0)

    def test_invalid_contact_frame_offset(self):
        with pytest.raises(ValueError, match="contact_frame_offset"):
            SwingAnalyzerConfig(contact_frame_offset=-1)

    def test_invalid_follow_through_offset(self):
        with pytest.raises(ValueError, match="follow_through_offset"):
            SwingAnalyzerConfig(follow_through_offset=1.5)

    def test_invalid_forward_swing_search_window(self):
        with pytest.raises(ValueError, match="forward_swing_search_window"):
            SwingAnalyzerConfig(forward_swing_search_window=0)

    def test_invalid_min_valid_frames(self):
        with pytest.raises(ValueError, match="min_valid_frames"):
            SwingAnalyzerConfig(min_valid_frames=0)

    def test_invalid_kinematic_chain_mode(self):
        with pytest.raises(ValueError, match="kinematic_chain_mode"):
            SwingAnalyzerConfig(kinematic_chain_mode="yes")

    def test_invalid_contact_detection_method(self):
        with pytest.raises(ValueError, match="contact_detection_method"):
            SwingAnalyzerConfig(contact_detection_method="invalid")

    def test_repr(self):
        config = SwingAnalyzerConfig()
        r = repr(config)
        assert "SwingAnalyzerConfig" in r
        assert "velocity_threshold=0.5" in r


class TestPresets:
    """Tests for preset configurations."""

    def test_standard_preset(self):
        assert PRESET_STANDARD.velocity_threshold == 0.5
        assert PRESET_STANDARD.contact_angle_min == 150

    def test_sensitive_preset(self):
        assert PRESET_SENSITIVE.velocity_threshold == 0.3
        assert PRESET_SENSITIVE.contact_angle_min == 120
        assert PRESET_SENSITIVE.use_adaptive_velocity is True

    def test_strict_preset(self):
        assert PRESET_STRICT.velocity_threshold == 0.7
        assert PRESET_STRICT.contact_angle_min == 160


class TestSwingAnalyzerInit:
    """Tests for SwingAnalyzer initialization."""

    def test_default_init(self):
        analyzer = SwingAnalyzer()
        assert analyzer.velocity_threshold == 0.5
        assert analyzer.kinematic_chain_mode is False

    def test_config_init(self):
        config = SwingAnalyzerConfig(velocity_threshold=0.8)
        analyzer = SwingAnalyzer(config=config)
        assert analyzer.velocity_threshold == 0.8

    def test_kwargs_init(self):
        analyzer = SwingAnalyzer(velocity_threshold=0.3, contact_angle_min=120)
        assert analyzer.velocity_threshold == 0.3
        assert analyzer.contact_angle_min == 120

    def test_preset_init(self):
        analyzer = SwingAnalyzer(config=PRESET_SENSITIVE)
        assert analyzer.velocity_threshold == 0.3


class TestAnalyzeSwing:
    """Tests for analyze_swing with synthetic video data."""

    def _make_video_data(self, num_frames=60, fps=30.0):
        """Create synthetic video data simulating a forehand swing."""
        frames = []
        for i in range(num_frames):
            t = i / fps
            # Simulate wrist moving: behind body (0-20), forward (20-40), past body (40-60)
            if i < 20:
                # Backswing phase - wrist moves behind body
                wrist_x = 0.5 - (i / 20) * 0.2  # 0.5 -> 0.3
                elbow_angle_approx = 90 + i * 2
            elif i < 40:
                # Forward swing - wrist moves forward fast
                progress = (i - 20) / 20
                wrist_x = 0.3 + progress * 0.5  # 0.3 -> 0.8
                elbow_angle_approx = 130 + progress * 40  # 130 -> 170
            else:
                # Follow through
                wrist_x = 0.8 + ((i - 40) / 20) * 0.1
                elbow_angle_approx = 160

            landmarks = {
                "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 0.9},
                "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0.0, "visibility": 0.9},
                "left_elbow": {"x": 0.35, "y": 0.4, "z": 0.0, "visibility": 0.9},
                "right_elbow": {"x": 0.65, "y": 0.4, "z": 0.0, "visibility": 0.9},
                "left_wrist": {"x": 0.35, "y": 0.5, "z": 0.0, "visibility": 0.9},
                "right_wrist": {"x": wrist_x, "y": 0.5, "z": 0.0, "visibility": 0.9},
                "left_hip": {"x": 0.4, "y": 0.5, "z": 0.0, "visibility": 0.9},
                "right_hip": {"x": 0.6, "y": 0.5, "z": 0.0, "visibility": 0.9},
            }

            frames.append({
                "frame_number": i + 1,
                "timestamp": t,
                "landmarks": landmarks,
                "pose_detected": True,
            })

        return {
            "fps": fps,
            "frame_count": num_frames,
            "width": 1920,
            "height": 1080,
            "frames": frames,
        }

    def test_returns_results_object(self):
        analyzer = SwingAnalyzer()
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        assert hasattr(results, "phases")
        assert hasattr(results, "engine")
        assert hasattr(results, "tempo")
        assert hasattr(results, "kinetic_chain")

    def test_detects_phases(self):
        analyzer = SwingAnalyzer(
            velocity_threshold=0.1,
            contact_angle_min=100,
        )
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        # Should detect at least backswing (wrist goes behind body)
        assert results.phases["unit_turn"] is not None
        if results.phases["unit_turn"]:
            assert results.phases["unit_turn"]["detected"] is True

    def test_insufficient_frames(self):
        analyzer = SwingAnalyzer(min_valid_frames=100)
        video_data = self._make_video_data(num_frames=10)
        results = analyzer.analyze_swing(video_data)
        # Should return empty results
        assert results.get_phases_detected_count() == 0

    def test_no_pose_detected(self):
        analyzer = SwingAnalyzer(min_valid_frames=5)
        video_data = {
            "fps": 30.0,
            "frame_count": 30,
            "width": 1920,
            "height": 1080,
            "frames": [
                {"frame_number": i, "timestamp": i / 30.0, "landmarks": None, "pose_detected": False}
                for i in range(30)
            ],
        }
        results = analyzer.analyze_swing(video_data)
        assert results.get_phases_detected_count() == 0

    def test_tracking_quality_passed_through(self):
        analyzer = SwingAnalyzer()
        video_data = self._make_video_data()
        video_data["tracking_quality"] = {
            "detection_rate": 0.95,
            "high_confidence_rate": 0.90,
            "average_confidence": 0.85,
        }
        results = analyzer.analyze_swing(video_data)
        assert results.tracking_quality is not None
        assert results.tracking_quality["detection_rate"] == 0.95

    def test_adaptive_velocity(self):
        analyzer = SwingAnalyzer(
            use_adaptive_velocity=True,
            adaptive_velocity_percent=0.15,
            contact_angle_min=100,
        )
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        # Should still produce results without error
        assert hasattr(results, "phases")

    def test_kinematic_chain_mode(self):
        analyzer = SwingAnalyzer(
            kinematic_chain_mode=True,
            contact_detection_method="kinematic_chain",
            velocity_threshold=0.1,
            contact_angle_min=100,
        )
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        assert hasattr(results, "phases")

    def test_hybrid_mode(self):
        analyzer = SwingAnalyzer(
            contact_detection_method="hybrid",
            velocity_threshold=0.1,
            contact_angle_min=100,
        )
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        assert hasattr(results, "phases")

    def test_engine_metrics_populated(self):
        analyzer = SwingAnalyzer(velocity_threshold=0.1, contact_angle_min=100)
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        engine = results.engine
        assert engine["hip_shoulder_separation"] is not None

    def test_tempo_metrics_with_detected_phases(self):
        analyzer = SwingAnalyzer(velocity_threshold=0.1, contact_angle_min=100)
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        tempo = results.tempo
        # If backswing and forward swing detected, backswing_duration should be set
        if (results.phases.get("unit_turn") and results.phases["unit_turn"].get("detected") and
            results.phases.get("forward_swing") and results.phases["forward_swing"].get("detected")):
            assert tempo["backswing_duration"] is not None

    def test_kinetic_chain_metrics_populated(self):
        analyzer = SwingAnalyzer(velocity_threshold=0.1, contact_angle_min=100)
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        kc = results.kinetic_chain
        assert kc["peak_velocity_sequence"] is not None

    def test_to_dict_output(self):
        analyzer = SwingAnalyzer(velocity_threshold=0.1, contact_angle_min=100)
        video_data = self._make_video_data()
        results = analyzer.analyze_swing(video_data)
        d = results.to_dict()
        assert "phases" in d
        assert "engine" in d
        assert "tempo" in d
        assert "kinetic_chain" in d
