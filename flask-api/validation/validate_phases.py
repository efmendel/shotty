"""
Offline phase detection validation tool.

Runs the swing analysis pipeline on local test videos and compares
detected phases against ground truth labels defined in ground_truth.json.
No Supabase or network access required.

Usage:
    cd flask-api
    python -m validation.validate_phases               # all videos
    python -m validation.validate_phases --video X.mp4  # single video
    python -m validation.validate_phases --verbose      # detailed output
"""

import argparse
import json
import logging
import os
import sys

from services.swing_analyzer import SwingAnalyzer
from services.video_processor import PRESET_DIFFICULT_VIDEO, VideoProcessor

logger = logging.getLogger(__name__)

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(VALIDATION_DIR, "videos")
GROUND_TRUTH_PATH = os.path.join(VALIDATION_DIR, "ground_truth.json")

PHASE_NAMES = ["unit_turn", "backswing", "forward_swing", "contact", "follow_through"]
DEFAULT_TOLERANCE = 5


def load_ground_truth():
    """
    Load ground truth phase labels from ground_truth.json.

    Returns:
        Dictionary mapping video filenames to expected phase data.

    Raises:
        FileNotFoundError: If ground_truth.json is missing.
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

    with open(GROUND_TRUTH_PATH, "r") as f:
        data = json.load(f)

    # Filter out metadata keys (e.g. _README)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def run_pipeline(video_path):
    """
    Run pose extraction and swing analysis on a local video file.

    Uses the same configuration as the production API endpoint.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        SwingAnalysisResults from the analyzer.
    """
    processor = VideoProcessor(pose_config=PRESET_DIFFICULT_VIDEO)
    video_data = processor.process_video(video_path)

    analyzer = SwingAnalyzer(
        use_adaptive_velocity=True,
        adaptive_velocity_percent=0.15,
        contact_angle_min=120,
        kinematic_chain_mode=True,
        contact_detection_method="hybrid",
    )

    return analyzer.analyze_swing(video_data)


def compare_phases(results, expected, tolerance):
    """
    Compare detected phases against ground truth.

    Args:
        results: SwingAnalysisResults from the analyzer.
        expected: Dict of expected phases from ground_truth.json.
        tolerance: Default frame tolerance for this video.

    Returns:
        List of dicts with comparison results per phase.
    """
    comparisons = []

    for phase_name in PHASE_NAMES:
        if phase_name not in expected:
            continue

        truth = expected[phase_name]
        expected_frame = truth["frame"]
        phase_tolerance = truth.get("tolerance", tolerance)

        detected = results.phases.get(phase_name)

        if detected is None or not detected.get("detected", False):
            comparisons.append({
                "phase": phase_name,
                "expected_frame": expected_frame,
                "detected_frame": None,
                "delta": None,
                "tolerance": phase_tolerance,
                "confidence": None,
                "passed": False,
                "reason": "not detected",
            })
            continue

        detected_frame = detected["frame"]
        delta = abs(detected_frame - expected_frame)
        passed = delta <= phase_tolerance

        comparisons.append({
            "phase": phase_name,
            "expected_frame": expected_frame,
            "detected_frame": detected_frame,
            "delta": delta,
            "tolerance": phase_tolerance,
            "confidence": detected.get("confidence", 0.0),
            "passed": passed,
            "reason": None,
        })

    return comparisons


def print_report(video_name, comparisons, verbose=False):
    """
    Print a formatted comparison report for a single video.

    Args:
        video_name: Filename of the video.
        comparisons: List of phase comparison dicts.
        verbose: Whether to print extra detail.
    """
    passed = sum(1 for c in comparisons if c["passed"])
    total = len(comparisons)

    status = "PASS" if passed == total else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {video_name}  [{status}] ({passed}/{total} phases)")
    print(f"{'='*60}")
    print(f"  {'Phase':<16} {'Expected':>8} {'Detected':>8} {'Delta':>6} {'Tol':>4}  Result")
    print(f"  {'-'*58}")

    for c in comparisons:
        detected_str = str(c["detected_frame"]) if c["detected_frame"] is not None else "---"
        delta_str = str(c["delta"]) if c["delta"] is not None else "---"
        result_str = "PASS" if c["passed"] else "FAIL"

        if c["reason"]:
            result_str += f" ({c['reason']})"

        conf_str = ""
        if verbose and c["confidence"] is not None:
            conf_str = f"  conf={c['confidence']:.2f}"

        print(
            f"  {c['phase']:<16} {c['expected_frame']:>8} {detected_str:>8} "
            f"{delta_str:>6} {c['tolerance']:>4}  {result_str}{conf_str}"
        )


def print_summary(all_results):
    """
    Print an overall summary across all validated videos.

    Args:
        all_results: List of (video_name, comparisons) tuples.
    """
    total_phases = 0
    total_passed = 0
    total_videos = len(all_results)
    videos_passed = 0

    for _, comparisons in all_results:
        video_pass = True
        for c in comparisons:
            total_phases += 1
            if c["passed"]:
                total_passed += 1
            else:
                video_pass = False
        if video_pass:
            videos_passed += 1

    total_failed = total_phases - total_passed
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Videos: {videos_passed}/{total_videos} passed")
    print(f"  Phases: {total_passed}/{total_phases} passed, {total_failed} failed")

    if total_phases > 0:
        rate = (total_passed / total_phases) * 100
        print(f"  Pass rate: {rate:.1f}%")

    print()


def main():
    """Run phase detection validation against ground truth labels."""
    parser = argparse.ArgumentParser(
        description="Validate swing phase detection against ground truth labels."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Run validation for a single video filename only.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed metrics (confidence scores, etc.).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # --- Load ground truth ---
    ground_truth = load_ground_truth()

    if not ground_truth:
        print("No entries in ground_truth.json (add videos and their expected phases).")
        sys.exit(1)

    # --- Filter to single video if requested ---
    if args.video:
        if args.video not in ground_truth:
            print(f"Video '{args.video}' not found in ground_truth.json.")
            sys.exit(1)
        ground_truth = {args.video: ground_truth[args.video]}

    # --- Validate each video ---
    all_results = []
    skipped = []

    for video_name, truth_data in ground_truth.items():
        video_path = os.path.join(VIDEOS_DIR, video_name)

        if not os.path.exists(video_path):
            skipped.append(video_name)
            continue

        print(f"\nProcessing {video_name}...")

        try:
            results = run_pipeline(video_path)
        except Exception as e:
            print(f"  ERROR: Pipeline failed for {video_name}: {e}")
            continue

        tolerance = truth_data.get("tolerance", DEFAULT_TOLERANCE)
        comparisons = compare_phases(results, truth_data["phases"], tolerance)
        print_report(video_name, comparisons, verbose=args.verbose)
        all_results.append((video_name, comparisons))

    # --- Print skipped videos ---
    if skipped:
        print(f"\nSkipped (video file not found in validation/videos/):")
        for name in skipped:
            print(f"  - {name}")

    # --- Print summary ---
    if all_results:
        print_summary(all_results)
    elif not skipped:
        print("\nNo videos to validate.")
    else:
        print("\nNo videos found. Place .mp4 files in validation/videos/.")

    # Exit with error code if any failures
    all_passed = all(
        c["passed"] for _, comparisons in all_results for c in comparisons
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
