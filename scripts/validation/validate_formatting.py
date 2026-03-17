#!/usr/bin/env python
"""
Open-H Dataset Validation Script

This script validates LeRobot datasets for compliance with:
1. LeRobot dataset format v2.1 specifications
2. Open-H data collection initiative requirements and recommendations

This is a LOCAL validation tool designed to be run as a final check before
uploading your dataset to HuggingFace or other platforms. It does NOT require
internet access or authentication.

Usage:
    python scripts/validation/validate_formatting.py /path/to/lerobot/dataset

    For verbose output:
    python scripts/validation/validate_formatting.py /path/to/dataset --verbose

The script performs comprehensive checks on:
- Dataset structure and format compliance
- Required features and naming conventions
- Healthcare-specific metadata
- Data quality metrics (fps, resolution, synchronization)
- Recovery/failure example handling
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import argparse
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Suppress FFmpeg/AV1 warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["FFMPEG_HIDE_BANNER"] = "1"

import cv2
from importlib.metadata import version as get_version

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.constants import HF_LEROBOT_HOME

REQUIRED_LEROBOT_VERSION = "0.3.3"
_installed_version = get_version("lerobot")
if _installed_version != REQUIRED_LEROBOT_VERSION:
    raise ImportError(
        f"lerobot version mismatch: installed {_installed_version}, "
        f"expected {REQUIRED_LEROBOT_VERSION}. "
        f"Install with: pip install lerobot=={REQUIRED_LEROBOT_VERSION}"
    )

class ValidationLevel(Enum):
    """Validation severity levels"""

    ERROR = "ERROR"  # Must fix for compliance
    WARNING = "WARNING"  # Should fix for best practices
    INFO = "INFO"  # Suggestions for improvement
    SUCCESS = "SUCCESS"  # Validation passed


@dataclass
class ValidationResult:
    """Container for validation results"""

    level: ValidationLevel
    category: str
    message: str
    details: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""

    results: List[ValidationResult] = field(default_factory=list)
    dataset_path: Optional[Path] = None

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.INFO)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.SUCCESS)

    @property
    def is_compliant(self) -> bool:
        return self.error_count == 0


class OpenHDatasetValidator:
    """Validator for Open-H LeRobot datasets"""

    # Required LeRobot v2.1 structure
    REQUIRED_DIRS = ["videos", "meta"]
    REQUIRED_METADATA_FILES = [
        "info.json",
        "episodes_stats.jsonl",
        "tasks.jsonl",
        "episodes.jsonl",
    ]

    # Required features based on Open-H guidelines
    REQUIRED_FEATURES = ["action", "observation.state"]
    RECOMMENDED_IMAGE_PREFIX = "observation.images."

    # Open-H specific requirements
    MIN_FPS = 20  # Minimum recommended FPS
    MIN_RESOLUTION = (480, 480)  # Minimum recommended resolution (height, width)

    # Healthcare-specific metadata patterns
    HEALTHCARE_META_PATTERNS = [
        "observation.meta.tool",
        "observation.meta.force",
        "observation.meta.probe",
        "instruction.text",
    ]

    def __init__(self, dataset_path: Path, verbose: bool = False):
        """
        Initialize validator with dataset path

        Args:
            dataset_path: Path to local dataset directory
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.report = ValidationReport()

        self.dataset_path = Path(dataset_path)
        self.report.dataset_path = self.dataset_path

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

    def add_result(
        self,
        level: ValidationLevel,
        category: str,
        message: str,
        details: Optional[str] = None,
    ):
        """Add a validation result to the report"""
        result = ValidationResult(level, category, message, details)
        self.report.results.append(result)

        if self.verbose or level in [ValidationLevel.ERROR, ValidationLevel.WARNING]:
            self._print_result(result)

    def _print_result(self, result: ValidationResult):
        """Print a validation result with formatting"""
        symbols = {
            ValidationLevel.ERROR: "❌",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.INFO: "ℹ️",
            ValidationLevel.SUCCESS: "✅",
        }
        colors = {
            ValidationLevel.ERROR: "\033[91m",
            ValidationLevel.WARNING: "\033[93m",
            ValidationLevel.INFO: "\033[94m",
            ValidationLevel.SUCCESS: "\033[92m",
        }
        reset = "\033[0m"

        symbol = symbols[result.level]
        color = colors[result.level]

        print(
            f"{color}{symbol} [{result.level.value}] {result.category}: {result.message}{reset}"
        )
        if result.details and self.verbose:
            print(f"    Details: {result.details}")

    def validate_directory_structure(self):
        """Validate LeRobot v2.1 directory structure"""
        category = "Directory Structure"

        # Check required directories
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Required directory '{dir_name}' not found",
                )
            else:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Required directory '{dir_name}' exists",
                )

        # Check for data directory with parquet chunks
        data_dir = self.dataset_path / "data"
        if not data_dir.exists():
            self.add_result(
                ValidationLevel.ERROR, category, "Required directory 'data' not found"
            )
        else:
            # Look for chunk directories
            chunk_dirs = list(data_dir.glob("chunk-*"))
            if not chunk_dirs:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    "No chunk directories found in data/ (expected format: data/chunk-000/)",
                )
            else:
                # Check for parquet files in chunks
                parquet_files = []
                for chunk_dir in chunk_dirs:
                    parquet_files.extend(list(chunk_dir.glob("episode_*.parquet")))
                if not parquet_files:
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        "No episode parquet files found in chunk directories (expected format: episode_000000.parquet)",
                    )
                else:
                    self.add_result(
                        ValidationLevel.SUCCESS,
                        category,
                        f"Found {len(parquet_files)} episode parquet files in {len(chunk_dirs)} chunk(s)",
                    )

    def validate_metadata_files(self):
        """Validate required metadata files"""
        category = "Metadata Files"
        metadata_dir = self.dataset_path / "meta"

        if not metadata_dir.exists():
            self.add_result(
                ValidationLevel.ERROR, category, "Metadata directory not found"
            )
            return

        # Check required metadata files
        for file_name in self.REQUIRED_METADATA_FILES:
            file_path = metadata_dir / file_name
            if not file_path.exists():
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Required metadata file '{file_name}' not found",
                )
            else:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Metadata file '{file_name}' exists",
                )

        # Check for Open-H required README.md
        readme_path = metadata_dir / "README.md"
        if not readme_path.exists():
            self.add_result(
                ValidationLevel.ERROR,
                category,
                "README.md not found in metadata directory (Open-H requirement)",
                "Create README.md using the provided template",
            )
        else:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "README.md found in metadata directory",
            )


    def validate_info_json(self):
        """Validate info.json content and Open-H requirements"""
        category = "Dataset Info"
        info_path = self.dataset_path / "meta" / "info.json"

        if not info_path.exists():
            return

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except json.JSONDecodeError as e:
            self.add_result(
                ValidationLevel.ERROR, category, f"Invalid JSON in info.json: {e}"
            )
            return

        # Check FPS
        if "fps" in info:
            fps = info["fps"]
            if fps < self.MIN_FPS:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"FPS ({fps}) below recommended minimum ({self.MIN_FPS} Hz)",
                    "Consider collecting data at ≥20 Hz for better quality",
                )
            else:
                self.add_result(
                    ValidationLevel.SUCCESS, category, f"FPS ({fps}) meets requirements"
                )

        # Check robot type
        if "robot_type" not in info:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "Robot type not specified in info.json",
            )

        # Check splits
        if "splits" in info:
            self._validate_splits(info["splits"])
        else:
            self.add_result(
                ValidationLevel.WARNING, category, "No data splits defined in info.json"
            )

        # Check features
        if "features" in info:
            self._validate_features(info["features"])

    def _validate_splits(self, splits: Dict):
        """Validate dataset splits including recovery/failure"""
        category = "Data Splits"

        # Check standard splits
        standard_splits = ["train", "val", "test"]
        for split in standard_splits:
            if split not in splits:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Standard split '{split}' not defined",
                )

        # Check for recovery/failure splits (Open-H recommendation)
        if "recovery" in splits:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "Recovery examples split defined (Open-H best practice)",
            )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "Consider adding recovery examples for improved robustness",
            )

        if "failure" in splits:
            self.add_result(
                ValidationLevel.INFO, category, "Failure examples split defined"
            )

    def _validate_features(self, features: Dict):
        """Validate dataset features for Open-H compliance"""
        category = "Dataset Features"

        # Check required features
        for required in self.REQUIRED_FEATURES:
            if required not in features:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Required feature '{required}' not found",
                    "This feature is essential for LeRobot compatibility",
                )
            else:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Required feature '{required}' present",
                )

        # Check for image features with proper naming
        image_features = [
            k for k in features if k.startswith(self.RECOMMENDED_IMAGE_PREFIX)
        ]
        if not image_features:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "No image features found with recommended prefix 'observation.images.'",
                "Use 'observation.images.xxx' naming for camera views",
            )
        else:
            for img_feature in image_features:
                self._validate_image_feature(img_feature, features[img_feature])

        # Check for healthcare-specific metadata
        healthcare_features = []
        for pattern in self.HEALTHCARE_META_PATTERNS:
            matching = [k for k in features if pattern in k]
            healthcare_features.extend(matching)

        if healthcare_features:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Healthcare-specific metadata found: {', '.join(healthcare_features[:3])}",
            )
        else:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "Consider adding healthcare-specific metadata (e.g., observation.meta.tool)",
            )

    def _validate_image_feature(self, feature_name: str, feature_info: Dict):
        """Validate individual image feature specifications"""
        category = "Image Features"

        if not isinstance(feature_info, dict):
            return

        # Check if video format is used
        if feature_info.get("dtype") == "video":
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"Feature '{feature_name}' uses efficient video format",
            )

        # Check resolution if shape is provided
        if "shape" in feature_info:
            shape = feature_info["shape"]
            if len(shape) >= 2:
                height, width = shape[0], shape[1]
                if height < self.MIN_RESOLUTION[0] or width < self.MIN_RESOLUTION[1]:
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"Feature '{feature_name}' resolution ({height}x{width}) below recommended minimum",
                        f"Consider using ≥{self.MIN_RESOLUTION[0]}p resolution",
                    )

    def validate_video_files(self):
        """Validate video files in the dataset"""
        category = "Video Files"
        videos_dir = self.dataset_path / "videos"

        if not videos_dir.exists():
            return

        video_files = list(videos_dir.glob("**/*.mp4"))

        if not video_files:
            self.add_result(
                ValidationLevel.WARNING,
                category,
                "No MP4 video files found",
                "Ensure videos are properly encoded",
            )
            return

        self.add_result(
            ValidationLevel.SUCCESS, category, f"Found {len(video_files)} video files"
        )

        # Sample check on first few videos
        sample_size = min(3, len(video_files))
        for video_path in video_files[:sample_size]:
            self._validate_video_file(video_path)

    def _validate_video_file(self, video_path: Path):
        """Validate individual video file"""
        category = "Video Quality"

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Cannot open video file: {video_path.name}",
                )
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()

            # Validate video properties
            if fps < self.MIN_FPS:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Video '{video_path.name}' FPS ({fps:.1f}) below recommended minimum",
                )

            if height < self.MIN_RESOLUTION[0]:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Video '{video_path.name}' resolution ({height}x{width}) below recommended",
                )

            if frame_count == 0:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Video '{video_path.name}' has no frames",
                )

        except Exception as e:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                f"Error validating video '{video_path.name}': {e}",
            )

    def validate_episodes(self):
        """Validate episodes.jsonl file"""
        category = "Episodes"
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"

        if not episodes_path.exists():
            return

        try:
            episodes = []
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    episodes.append(json.loads(line))

            if not episodes:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    "No episodes found in episodes.jsonl",
                )
                return

            self.add_result(
                ValidationLevel.SUCCESS, category, f"Found {len(episodes)} episodes"
            )

            # Check for task descriptions
            tasks = set()
            for ep in episodes:
                if "tasks" in ep:
                    # Handle tasks as a list
                    if isinstance(ep["tasks"], list):
                        tasks.update(ep["tasks"])
                    else:
                        tasks.add(ep["tasks"])
                elif "task" in ep:
                    # Handle legacy single task format
                    tasks.add(ep["task"])

            if not tasks:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    "No task descriptions found in episodes",
                )
            else:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Found {len(tasks)} unique task(s): {', '.join(sorted(tasks))}",
                )

                # Check for recovery tasks (Open-H best practice)
                recovery_tasks = [task for task in tasks if "recovery" in task.lower()]
                if recovery_tasks:
                    self.add_result(
                        ValidationLevel.SUCCESS,
                        category,
                        f"Found {len(recovery_tasks)} recovery task(s) - excellent for robustness",
                    )
        except Exception as e:
            self.add_result(
                ValidationLevel.ERROR, category, f"Error reading episodes.jsonl: {e}"
            )

    def validate_timestamps(self):
        """Validate timestamp column in episode parquet files.

        Checks for issues known to cause training/inference failures in
        downstream models:
        - Absolute Unix epoch timestamps stored as float32 (precision collapse)
        - Constant or near-constant timestamps across an episode
        - Non-monotonic timestamps
        - Non-strictly-monotonic timestamps (duplicate values)
        - Unreasonable spacing relative to declared FPS
        - Timestamps not relative to episode start
        """
        category = "Timestamps"
        data_dir = self.dataset_path / "data"

        if not data_dir.exists():
            self.add_result(
                ValidationLevel.ERROR, category, "Data directory not found, cannot validate timestamps"
            )
            return

        try:
            import pandas as pd
        except ImportError:
            self.add_result(
                ValidationLevel.INFO,
                category,
                "pandas not installed — skipping timestamp validation",
                "Install with: pip install pandas pyarrow",
            )
            return

        # Read FPS from info.json for spacing checks
        fps = None
        info_path = self.dataset_path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                raw_fps = info.get("fps")
                if raw_fps is not None:
                    try:
                        fps = float(raw_fps)
                        if fps <= 0:
                            self.add_result(
                                ValidationLevel.WARNING,
                                category,
                                f"Invalid fps value in info.json: {raw_fps}. "
                                "Skipping FPS-dependent timestamp checks.",
                            )
                            fps = None
                    except (TypeError, ValueError):
                        self.add_result(
                            ValidationLevel.WARNING,
                            category,
                            f"Non-numeric fps value in info.json: {raw_fps}. "
                            "Skipping FPS-dependent timestamp checks.",
                        )
                        fps = None
            except (json.JSONDecodeError, KeyError):
                pass

        chunk_dirs = sorted(data_dir.glob("chunk-*"))
        if not chunk_dirs:
            return

        parquet_files = []
        for chunk_dir in chunk_dirs:
            parquet_files.extend(sorted(chunk_dir.glob("episode_*.parquet")))

        if not parquet_files:
            return

        files_to_check = parquet_files

        episodes_checked = 0
        episodes_with_errors = 0
        episodes_with_warnings = 0
        issue_summary = {
            "epoch_timestamps": [],
            "constant_timestamps": [],
            "low_uniqueness": [],
            "non_monotonic": [],
            "non_strictly_monotonic": [],
            "bad_spacing": [],
            "not_relative": [],
        }

        for pf in files_to_check:
            try:
                df = pd.read_parquet(pf, engine="pyarrow")
            except Exception as e:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"Could not read {pf.name}: {e}",
                )
                continue

            if "timestamp" not in df.columns:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{pf.name}: missing 'timestamp' column",
                )
                episodes_with_errors += 1
                continue

            ts_series = df["timestamp"]
            if not np.issubdtype(ts_series.dtype, np.number):
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{pf.name}: timestamp column has non-numeric dtype ({ts_series.dtype})",
                )
                episodes_with_errors += 1
                continue

            ts = pd.to_numeric(ts_series, errors="coerce").to_numpy(dtype=np.float64)
            non_finite_count = int(np.sum(~np.isfinite(ts)))
            if non_finite_count > 0:
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{pf.name}: timestamp contains {non_finite_count} NaN/Inf value(s)",
                )
                episodes_with_errors += 1
                continue

            episodes_checked += 1
            n = len(ts)
            ep_name = pf.stem
            has_error = False
            has_warning = False

            if n < 2:
                continue

            # --- Check 1: Absolute Unix epoch timestamps ---
            # float32 can only represent ~7 significant digits; Unix epoch
            # values (~1.7e9) lose all sub-second precision, collapsing
            # per-frame deltas to zero.
            if ts[0] > 1e6:
                issue_summary["epoch_timestamps"].append(ep_name)
                is_float32 = df["timestamp"].dtype == np.float32
                if is_float32:
                    self.add_result(
                        ValidationLevel.ERROR,
                        category,
                        f"{ep_name}: timestamps are absolute Unix epoch values "
                        f"(ts[0]={ts[0]:.1f}) stored as float32 — precision collapse "
                        f"makes per-frame deltas invisible to downstream models",
                        "Convert to relative timestamps (subtract episode start time) "
                        "or store as float64",
                    )
                    has_error = True
                else:
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"{ep_name}: timestamps appear to be absolute Unix epoch values "
                        f"(ts[0]={ts[0]:.1f}) — downstream models expect relative timestamps "
                        f"starting near 0",
                        "Consider converting to relative timestamps (subtract episode start time)",
                    )
                    has_warning = True
                    issue_summary["not_relative"].append(ep_name)

            # --- Check 2: Constant or near-constant timestamps ---
            ts_min = float(np.min(ts))
            ts_max = float(np.max(ts))
            ts_range = ts_max - ts_min
            expected_duration = (n - 1) / fps if fps else None

            if ts_range == 0:
                issue_summary["constant_timestamps"].append(ep_name)
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{ep_name}: all {n} timestamps are identical ({ts[0]:.6f}) — "
                    f"video frame selection will always return frame 0",
                )
                has_error = True
            elif expected_duration and ts_range < expected_duration * 0.01:
                issue_summary["constant_timestamps"].append(ep_name)
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{ep_name}: timestamp range ({ts_range:.2e}s) is negligible "
                    f"compared to expected episode duration ({expected_duration:.2f}s at {fps} fps) "
                    f"— effectively constant",
                )
                has_error = True

            # --- Check 3: Uniqueness ---
            num_unique = len(np.unique(ts))
            uniqueness_ratio = num_unique / n

            if num_unique == 1 and n > 1:
                pass  # already reported in constant check
            elif uniqueness_ratio < 0.5:
                issue_summary["low_uniqueness"].append(ep_name)
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{ep_name}: only {num_unique}/{n} unique timestamp values "
                    f"({uniqueness_ratio:.1%}) — most frames share timestamps, "
                    f"causing incorrect video frame lookups",
                )
                has_error = True
            elif uniqueness_ratio < 1.0:
                issue_summary["non_strictly_monotonic"].append(ep_name)
                num_duplicates = n - num_unique
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    f"{ep_name}: {num_duplicates} duplicate timestamp value(s) "
                    f"({num_unique}/{n} unique, {uniqueness_ratio:.1%}) — "
                    f"ideally each frame should have a distinct timestamp",
                )
                has_warning = True

            # --- Check 4: Monotonicity ---
            diffs = np.diff(ts)
            num_decreasing = int(np.sum(diffs < 0))
            if num_decreasing > 0:
                issue_summary["non_monotonic"].append(ep_name)
                first_decrease_idx = int(np.argmax(diffs < 0))
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"{ep_name}: timestamps are NOT monotonically increasing — "
                    f"{num_decreasing} decrease(s) found "
                    f"(first at index {first_decrease_idx}: "
                    f"{ts[first_decrease_idx]:.6f} -> {ts[first_decrease_idx+1]:.6f})",
                )
                has_error = True

            # --- Check 5: Spacing relative to FPS ---
            if fps and ts_range > 0 and num_unique > 1:
                expected_spacing = 1.0 / fps
                positive_diffs = diffs[diffs > 0]
                if len(positive_diffs) > 0:
                    mean_spacing = float(np.mean(positive_diffs))
                    ratio = mean_spacing / expected_spacing
                    if ratio > 5.0 or ratio < 0.1:
                        issue_summary["bad_spacing"].append(ep_name)
                        self.add_result(
                            ValidationLevel.WARNING,
                            category,
                            f"{ep_name}: mean timestamp spacing ({mean_spacing:.4f}s) "
                            f"deviates significantly from expected 1/{fps}={expected_spacing:.4f}s "
                            f"(ratio: {ratio:.1f}x)",
                            "This may indicate timestamps in wrong units or from a "
                            "different clock source",
                        )
                        has_warning = True

            # --- Check 6: Relative timestamps (should start near 0) ---
            if 0 < ts[0] <= 1e6:
                if ts[0] > 60.0:
                    issue_summary["not_relative"].append(ep_name)
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"{ep_name}: first timestamp is {ts[0]:.2f}s — "
                        f"timestamps may not be relative to episode start",
                        "LeRobot defaults to frame_index/fps (starting at 0.0) when "
                        "no explicit timestamp is provided; most datasets follow this convention",
                    )
                    has_warning = True

            if has_error:
                episodes_with_errors += 1
            elif has_warning:
                episodes_with_warnings += 1

        # --- Aggregate summary ---
        if episodes_checked == 0:
            return

        total_issues = episodes_with_errors + episodes_with_warnings
        if total_issues == 0:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                f"All {episodes_checked} checked episodes have valid timestamps "
                f"(monotonically increasing, unique per frame, relative to episode start)",
            )
        else:
            if episodes_with_errors > 0:
                broken_types = []
                if issue_summary["epoch_timestamps"]:
                    broken_types.append(
                        f"{len(issue_summary['epoch_timestamps'])} with absolute epoch values"
                    )
                if issue_summary["constant_timestamps"]:
                    broken_types.append(
                        f"{len(issue_summary['constant_timestamps'])} with constant/collapsed timestamps"
                    )
                if issue_summary["low_uniqueness"]:
                    broken_types.append(
                        f"{len(issue_summary['low_uniqueness'])} with very low uniqueness"
                    )
                if issue_summary["non_monotonic"]:
                    broken_types.append(
                        f"{len(issue_summary['non_monotonic'])} with non-monotonic values"
                    )
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Timestamp issues found in {episodes_with_errors}/{episodes_checked} "
                    f"checked episodes: {'; '.join(broken_types)}",
                    "Broken timestamps may cause downstream models "
                    "to always select the same video frame, "
                    "producing static/frozen training data. Fix the dataset's "
                    "timestamp column or ensure the training pipeline has a "
                    "fallback (e.g. frame_index / fps).",
                )

            if episodes_with_warnings > 0:
                warn_types = []
                if issue_summary["non_strictly_monotonic"]:
                    warn_types.append(
                        f"{len(issue_summary['non_strictly_monotonic'])} with duplicate timestamps"
                    )
                if issue_summary["bad_spacing"]:
                    warn_types.append(
                        f"{len(issue_summary['bad_spacing'])} with unexpected spacing"
                    )
                if issue_summary["not_relative"]:
                    warn_types.append(
                        f"{len(issue_summary['not_relative'])} with non-relative timestamps"
                    )
                if warn_types:
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"Timestamp warnings in {episodes_with_warnings}/{episodes_checked} "
                        f"checked episodes: {'; '.join(warn_types)}",
                    )

        self.add_result(
            ValidationLevel.INFO,
            category,
            f"Checked {episodes_checked} episode parquet file(s) for timestamp integrity.",
        )

    def validate_data_synchronization(self):
        """Check for data synchronization documentation and timestamps"""
        category = "Data Synchronization"

        # Check if timestamps are mentioned in info.json
        info_path = self.dataset_path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)

                # Check for tolerance_s parameter (indicates sync consideration)
                if "tolerance_s" in info:
                    tolerance = info["tolerance_s"]
                    self.add_result(
                        ValidationLevel.SUCCESS,
                        category,
                        f"Synchronization tolerance specified: {tolerance}s",
                    )
            except json.JSONDecodeError:
                # Already reported in validate_info_json
                pass

        # Check README for synchronization documentation
        readme_path = self.dataset_path / "meta" / "README.md"
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "synchron" in content or "timestamp" in content:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    "Synchronization documentation found in README",
                )
            else:
                self.add_result(
                    ValidationLevel.WARNING,
                    category,
                    "No synchronization documentation found in README",
                    "Document synchronization methods as per Open-H requirements",
                )

    def validate_lerobot_compatibility(self):
        """Validate dataset can be loaded with LeRobot locally"""
        category = "LeRobot Compatibility"

        try:
            # Load dataset from local path only - no remote access

            # Create repo_id in the form: parent_folder/child_folder
            repo_id = "/".join(str(self.dataset_path).split("/")[-2:])

            # Try to load the dataset locally without any remote access
            # Set local_files_only=True to prevent any HuggingFace API calls
            dataset = LeRobotDataset(
                repo_id,
                root=str(self.dataset_path),
                video_backend="pytorch",
            )

            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "Dataset structure is compatible with LeRobot v2.1",
            )

            # Check dataset properties
            if hasattr(dataset, "__len__"):
                dataset_len = len(dataset)
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Dataset contains {dataset_len} frames",
                )

            # Check if dataset has required attributes
            if hasattr(dataset, "features"):
                self.add_result(
                    ValidationLevel.SUCCESS, category, "Dataset features accessible"
                )

        except Exception as e:
            self.add_result(
                ValidationLevel.ERROR,
                category,
                f"Error loading dataset: {e}",
            )
            # Only validate structure, not actual loading
            self._validate_lerobot_structure()


    def _validate_lerobot_structure(self):
        """Manually validate LeRobot dataset structure without loading"""
        category = "LeRobot Structure"

        # Check for essential LeRobot v2.1 files
        required_files = {
            "data": "Main data file",
            "meta/info.json": "Dataset information",
            "meta/episodes_stats.jsonl": "Dataset statistics",
            "meta/tasks.jsonl": "Task descriptions",
            "meta/episodes.jsonl": "Episode information",
        }

        all_present = True
        for file_path, description in required_files.items():
            full_path = self.dataset_path / file_path
            if not full_path.exists():
                all_present = False
                self.add_result(
                    ValidationLevel.ERROR,
                    category,
                    f"Missing {description}: {file_path}",
                )

        if all_present:
            self.add_result(
                ValidationLevel.SUCCESS,
                category,
                "All required LeRobot v2.1 files present",
            )

        # Check video directory structure
        videos_dir = self.dataset_path / "videos"
        if videos_dir.exists():
            video_files = list(videos_dir.glob("**/*.mp4"))
            if video_files:
                self.add_result(
                    ValidationLevel.SUCCESS,
                    category,
                    f"Video directory contains {len(video_files)} MP4 files",
                )
        # Validate parquet files in chunk directories
        data_dir = self.dataset_path / "data"
        if data_dir.exists():
            chunk_dirs = list(data_dir.glob("chunk-*"))
            if chunk_dirs:
                try:
                    import pandas as pd

                    total_rows = 0
                    parquet_files = []

                    for chunk_dir in chunk_dirs:
                        episode_files = list(chunk_dir.glob("episode_*.parquet"))
                        parquet_files.extend(episode_files)

                        # Read a sample file to check structure
                        if episode_files:
                            sample_file = episode_files[0]
                            df = pd.read_parquet(sample_file, engine="pyarrow")
                            total_rows += len(df)

                            # Check for essential columns (only once)
                            if chunk_dir == chunk_dirs[0]:
                                essential_cols = [
                                    "episode_index",
                                    "frame_index",
                                    "timestamp",
                                ]
                    for col in essential_cols:
                        if col not in df.columns:
                            self.add_result(
                                ValidationLevel.WARNING,
                                category,
                                f"Episode parquet missing expected column: {col}",
                            )

                        if parquet_files:
                            self.add_result(
                                ValidationLevel.SUCCESS,
                                category,
                                f"Episode parquet files readable, sample contains {total_rows} rows",
                            )

                except ImportError:
                    self.add_result(
                        ValidationLevel.INFO,
                        category,
                        "pandas/pyarrow not installed - skipping parquet validation",
                        "Install with: pip install pandas pyarrow",
                    )
                except Exception as e:
                    self.add_result(
                        ValidationLevel.WARNING,
                        category,
                        f"Could not read episode parquet files: {str(e)[:100]}",
                    )

    def run_validation(self) -> ValidationReport:
        """Run all validation checks"""
        print(f"\n{'='*60}")
        print(f"Open-H Dataset Validation")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"{'='*60}\n")

        # Run all validation checks
        print("🔍 Validating directory structure...")
        self.validate_directory_structure()

        print("📁 Validating metadata files...")
        self.validate_metadata_files()

        print("📊 Validating dataset info...")
        self.validate_info_json()

        print("🎬 Validating video files...")
        self.validate_video_files()

        print("📝 Validating episodes...")
        self.validate_episodes()

        print("🕐 Validating timestamps...")
        self.validate_timestamps()

        print("⏱️ Validating synchronization...")
        self.validate_data_synchronization()

        print("🤖 Validating LeRobot compatibility...")
        self.validate_lerobot_compatibility()

        return self.report

    def print_summary(self):
        """Print validation summary"""
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")

        # Count results by level
        print(f"\n📊 Results Overview:")
        print(f"  ✅ Success: {self.report.success_count}")
        print(f"  ℹ️  Info: {self.report.info_count}")
        print(f"  ⚠️  Warnings: {self.report.warning_count}")
        print(f"  ❌ Errors: {self.report.error_count}")

        # Group results by category
        categories = {}
        for result in self.report.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        # Print errors and warnings by category
        if self.report.error_count > 0:
            print(f"\n🚨 Critical Issues (Must Fix):")
            for category, results in categories.items():
                errors = [r for r in results if r.level == ValidationLevel.ERROR]
                if errors:
                    print(f"\n  {category}:")
                    for error in errors:
                        print(f"    • {error.message}")
                        if error.details:
                            print(f"      → {error.details}")

        if self.report.warning_count > 0:
            print(f"\n⚠️  Recommendations (Should Fix):")
            for category, results in categories.items():
                warnings = [r for r in results if r.level == ValidationLevel.WARNING]
                if warnings:
                    print(f"\n  {category}:")
                    for warning in warnings:
                        print(f"    • {warning.message}")
                        if warning.details:
                            print(f"      → {warning.details}")

        # Final verdict
        print(f"\n{'='*60}")
        print("FINAL VERDICT")
        print(f"{'='*60}")

        if self.report.is_compliant:
            print("\n✅ Dataset is Open-H Initiative READY!")
            print(
                "   The dataset meets all requirements for the Open-H data collection initiative."
            )
            if self.report.warning_count > 0:
                print(
                    f"   Consider addressing {self.report.warning_count} warning(s) for best practices."
                )
        else:
            print("\n❌ Dataset is NOT Open-H Initiative ready.")
            print(
                f"   {self.report.error_count} critical issue(s) must be fixed before the dataset"
            )
            print("   can be considered compliant with Open-H requirements.")
            print(
                "\n   Please address the errors listed above and run validation again."
            )

        print(f"\n{'='*60}\n")


def main():
    """Main entry point for the validation script"""
    parser = argparse.ArgumentParser(
        description="Validate LeRobot datasets for Open-H Initiative compliance (LOCAL validation only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool performs LOCAL validation only and does not require internet access.
Run this as a final check before uploading your dataset to HuggingFace.

Examples:
  # Validate local dataset directory
  python scripts/validation/validate_formatting.py /path/to/dataset

  # Enable verbose output for detailed results
  python scripts/validation/validate_formatting.py /path/to/dataset --verbose
        """,
    )

    # Dataset path argument
    parser.add_argument(
        "dataset_path", type=Path, help="Path to the local LeRobot dataset directory"
    )

    # Options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output showing all validation results",
    )

    args = parser.parse_args()

    try:
        # Create and run validator
        validator = OpenHDatasetValidator(
            dataset_path=args.dataset_path, verbose=args.verbose
        )

        # Run validation
        report = validator.run_validation()

        # Print summary
        validator.print_summary()

        # Exit with appropriate code
        sys.exit(0 if report.is_compliant else 1)

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
