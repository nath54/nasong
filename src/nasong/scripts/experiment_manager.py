# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Experiment lifecycle management.

This module provides the `Experiment` and `ExperimentManager` classes for
tracking, saving, and loading training experiments. It handles metadata
storage and parameter persistence.
"""

#
### Import Modules. ###
#
from typing import Any, Optional

#
import os
import json
import time
import shutil
import uuid
from datetime import datetime

#
EXPERIMENTS_DIR = os.path.expanduser("~/.nasong/experiments")


class Experiment:
    """Represents a single NaSong training experiment.

    An experiment encapsulates the history, configuration, and artifacts of a
    specific training run.

    Attributes:
        id (str): Unique identifier for the experiment.
        name (str): Human-readable name (often the instrument name).
        timestamp (float): Creation time in Unix epoch.
        metrics (dict[str, Any]): Dictionary of performance results (e.g., loss).
        params (dict[str, Any]): Configuration parameters used for training.
        status (str): Current state (e.g., 'created', 'running', 'completed').
    """

    def __init__(
        self,
        experiment_id: str,
        name: str,
        timestamp: float,
        metrics: dict[str, Any],
        params: dict[str, Any],
        status: str = "created",
    ) -> None:
        """Initializes an Experiment object.

        Args:
            experiment_id (str): Unique hash or string ID.
            name (str): Label for the experiment.
            timestamp (float): Start time.
            metrics (dict[str, Any]): Initial metrics.
            params (dict[str, Any]): Hyperparameters.
            status (str, optional): Run status. Defaults to "created".
        """
        self.id = experiment_id
        self.name = name
        self.timestamp = timestamp
        self.metrics = metrics
        self.params = params  # Hyperparams/Config
        self.status = status

    @property
    def path(self) -> str:
        """Calculates the absolute file system path to the experiment folder.

        Returns:
            str: The full path to the experiment directory.
        """
        # Folder name convention: timestamp_name_id
        # We need to find the actual folder since timestamp might have slight precision diffs if we reconstruct
        # But simpler: we assume manager handles paths
        return os.path.join(EXPERIMENTS_DIR, f"{self.timestamp}_{self.name}_{self.id}")

    def save_meta(self) -> None:
        """Saves experiment metadata to 'meta.json' in the experiment path."""
        os.makedirs(self.path, exist_ok=True)
        meta = {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "params": self.params,
            "status": self.status,
            "date": datetime.fromtimestamp(self.timestamp).isoformat(),
        }
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def save_parameters_json(self, parameters: dict[str, float]) -> None:
        """Saves the trained instrument parameters for inference.

        Args:
            parameters (dict[str, float]): Dictionary of named parameter values.
        """
        with open(os.path.join(self.path, "params.json"), "w", encoding="utf-8") as f:
            json.dump(parameters, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Experiment":
        """Loads an existing experiment from its directory.

        Args:
            path (str): The folder path containing 'meta.json'.

        Returns:
            Experiment: The loaded experiment object.

        Raises:
            FileNotFoundError: If 'meta.json' is missing.
        """
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No experiment found at {path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return cls(
            experiment_id=meta["id"],
            name=meta["name"],
            timestamp=meta["timestamp"],
            metrics=meta.get("metrics", {}),
            params=meta.get("params", {}),
            status=meta.get("status", "unknown"),
        )


class ExperimentManager:
    """Entry point for managing and discovering multiple experiments.

    Attributes:
        base_dir (str): Root directory where all experiments are stored.
    """

    def __init__(self, base_dir: str = EXPERIMENTS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_experiment(self, name: str, params: dict[str, Any] = None) -> Experiment:
        """Creates and persists a new experiment.

        Args:
            name (str): Name for the experiment.
            params (dict[str, Any], optional): Configuration parameters.
                Defaults to None.

        Returns:
            Experiment: The newly created experiment.
        """
        experiment_id = str(uuid.uuid4())[:8]
        timestamp = time.time()
        exp = Experiment(
            experiment_id=experiment_id,
            name=name,
            timestamp=timestamp,
            metrics={},
            params=params or {},
            status="running",
        )
        exp.save_meta()
        return exp

    def list_experiments(self) -> list[Experiment]:
        """Lists all experiments found in the base directory.

        Returns:
            list[Experiment]: List of experiments, sorted by newest first.
        """
        experiments = []
        if not os.path.exists(self.base_dir):
            return []

        for dirname in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, dirname)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "meta.json")):
                try:
                    experiments.append(Experiment.load(path))
                except Exception:  # pylint: disable=broad-except
                    continue  # Skip corrupted

        # Sort by timestamp desc
        experiments.sort(key=lambda x: x.timestamp, reverse=True)
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieves a specific experiment by its ID or directory name.

        Args:
            experiment_id (str): The unique ID or folder name.

        Returns:
            Optional[Experiment]: The experiment if found, else None.
        """
        # Search by ID (since directory has timestamp)
        for dirname in os.listdir(self.base_dir):
            if (
                experiment_id in dirname
            ):  # Simple heuristics, better is to check ID inside
                path = os.path.join(self.base_dir, dirname)
                try:
                    exp = Experiment.load(path)
                    if (
                        exp.id == experiment_id or experiment_id == dirname
                    ):  # Support full dir name match too
                        return exp
                except Exception as _e:
                    pass
        return None

    def delete_experiment(self, experiment_id: str) -> bool:
        """Deletes an experiment and its associated files.

        Args:
            experiment_id (str): The identifier for the experiment to remove.

        Returns:
            bool: True if deleted successfully, False if not found.
        """
        exp = self.get_experiment(experiment_id)
        if exp:
            shutil.rmtree(exp.path)
            return True
        return False
