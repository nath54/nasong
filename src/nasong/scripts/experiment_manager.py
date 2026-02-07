import os
import json
import time
import shutil
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

EXPERIMENTS_DIR = os.path.expanduser("~/.nasong/experiments")


class Experiment:
    def __init__(
        self,
        experiment_id: str,
        name: str,
        timestamp: float,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        status: str = "created",
    ):
        self.id = experiment_id
        self.name = name
        self.timestamp = timestamp
        self.metrics = metrics
        self.params = params  # Hyperparams/Config
        self.status = status

    @property
    def path(self):
        # Folder name convention: timestamp_name_id
        # We need to find the actual folder since timestamp might have slight precision diffs if we reconstruct
        # But simpler: we assume manager handles paths
        return os.path.join(EXPERIMENTS_DIR, f"{self.timestamp}_{self.name}_{self.id}")

    def save_meta(self):
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
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def save_parameters_json(self, parameters: Dict[str, float]):
        """Save the trained instrument parameters for inference."""
        with open(os.path.join(self.path, "params.json"), "w") as f:
            json.dump(parameters, f, indent=2)

    @classmethod
    def load(cls, path: str):
        meta_path = os.path.join(path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No experiment found at {path}")

        with open(meta_path, "r") as f:
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
    def __init__(self, base_dir: str = EXPERIMENTS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_experiment(self, name: str, params: Dict[str, Any] = None) -> Experiment:
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

    def list_experiments(self) -> List[Experiment]:
        experiments = []
        if not os.path.exists(self.base_dir):
            return []

        for dirname in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, dirname)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "meta.json")):
                try:
                    experiments.append(Experiment.load(path))
                except Exception:
                    continue  # Skip corrupted

        # Sort by timestamp desc
        experiments.sort(key=lambda x: x.timestamp, reverse=True)
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
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
                except:
                    pass
        return None

    def delete_experiment(self, experiment_id: str) -> bool:
        exp = self.get_experiment(experiment_id)
        if exp:
            shutil.rmtree(exp.path)
            return True
        return False
