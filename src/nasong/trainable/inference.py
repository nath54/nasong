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
        exp = Experiment.load(experiment_id_or_path)
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
