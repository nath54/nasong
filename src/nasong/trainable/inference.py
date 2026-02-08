import os
import json
import functools
from typing import Callable
from nasong.scripts.experiment_manager import ExperimentManager, Experiment
from nasong.trainable.extract import get_trainable_instrument
from nasong.core.value import ParameterContext


def load_trained_instrument(experiment_id_or_path: str) -> Callable:
    """
    Load a trained instrument from an experiment and return a usable function.

    This function wraps the instrument blueprint with a ParameterContext,
    so that when called, it automatically uses the trained parameters
    (in pure Python/NumPy inference mode) instead of defaults.

    Args:
        experiment_id_or_path: ID of the experiment or full path to experiment folder.

    Returns:
        A callable function corresponding to the instrument, with trained parameters pre-loaded.
    """

    manager = ExperimentManager()

    # Resolve experiment
    if os.path.exists(experiment_id_or_path) and os.path.isdir(experiment_id_or_path):
        try:
            exp = Experiment.load(experiment_id_or_path)
        except Exception:
            # Fallback: try to construct experiment from config.yaml if meta.json is missing
            config_path = os.path.join(experiment_id_or_path, "config.yaml")
            if os.path.exists(config_path):
                # Import here to avoid circular dependencies if any
                import yaml

                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)

                # Create a dummy/wrapper experiment object
                exp = Experiment(
                    experiment_id="local",
                    name=config_data.get("instrument_name", "unknown"),
                    timestamp=0,
                    metrics={},
                    params={
                        "instrument": config_data.get("instrument_name", "unknown")
                    },
                    status="completed",
                )
                # Monkey-patch path
                original_path_prop = Experiment.path
                exp.__dict__["path"] = experiment_id_or_path
                # We need to ensure exp.path returns the directory.
                # Since Experiment.path is a property based on ID/timestamp usually,
                # we might need to override it or just set it if we control the class.
                # But Experiment class property calculates it.
                # Let's simple create a mock object or use the fact that we need exp.path later.

                class MockExperiment:
                    def __init__(self, path, params):
                        self.path = path
                        self.params = params
                        self.id = "local"

                exp = MockExperiment(
                    experiment_id_or_path,
                    {"instrument": config_data.get("instrument_name", "unknown")},
                )

            else:
                raise ValueError(
                    f"Could not load experiment from {experiment_id_or_path}: missing meta.json and config.yaml"
                )
    else:
        exp = manager.get_experiment(experiment_id_or_path)

    if not exp:
        raise ValueError(f"Could not find experiment: {experiment_id_or_path}")

    # Load parameters
    params_path = os.path.join(exp.path, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Experiment {exp.id} has no params.json (was training successful?)"
        )

    with open(params_path, "r") as f:
        trained_params = json.load(f)

    # Get instrument blueprint
    instrument_name = exp.params.get("instrument")
    if not instrument_name:
        raise ValueError("Experiment metadata missing 'instrument' name.")

    original_blueprint = get_trainable_instrument(instrument_name)

    # Create wrapper
    @functools.wraps(original_blueprint)
    def wrapper(*args, **kwargs):
        # Inject parameters into context during execution
        with ParameterContext(parameters=trained_params):
            return original_blueprint(*args, **kwargs)

    return wrapper
