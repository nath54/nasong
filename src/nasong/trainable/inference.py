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


"""Inference utilities for trained instruments.

This module provides tools for loading saved experiment results (trained
parameters) and wrapping the original instrument blueprints so they can be
easily used in sequencing and playback without manual parameter injection.
"""

#
### Import Modules. ###
#
from typing import Callable, Any

#
import os
import json
import functools

#
from nasong.scripts.experiment_manager import ExperimentManager, Experiment
from nasong.trainable.extract import get_trainable_instrument
from nasong.core.value import ParameterContext


#
def load_trained_instrument(experiment_id_or_path: str) -> Callable:
    """Loads a trained instrument and returns a pre-configured inference wrapper.

    The returned callable automatically uses the optimized parameters from the
    specified experiment. It uses `ParameterContext` internally to inject
    these values during the audio graph construction.

    Args:
        experiment_id_or_path (str): The ID of a completed experiment or the
            full file system path to the experiment directory.

    Returns:
        Callable: A function that takes the same arguments as the original
            instrument blueprint but uses trained parameter values.

    Raises:
        ValueError: If the experiment cannot be found or meta-data is corrupt.
        FileNotFoundError: If `params.json` is missing from the experiment.
    """

    manager = ExperimentManager()

    # Resolve experiment
    if os.path.exists(experiment_id_or_path) and os.path.isdir(experiment_id_or_path):
        try:
            exp: Any = Experiment.load(experiment_id_or_path)
        except Exception as e:  # pylint: disable=broad-except
            # Fallback: try to construct experiment from config.yaml if meta.json is missing
            config_path = os.path.join(experiment_id_or_path, "config.yaml")
            if os.path.exists(config_path):
                # Import here to avoid circular dependencies if any
                import yaml  # type: ignore  # pylint: disable=import-outside-toplevel

                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)

                class MockExperiment:
                    """
                    Mock experiment class for backward compatibility.
                    """

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
                    f"Could not load experiment from {experiment_id_or_path}:\
                      missing meta.json and config.yaml"
                ) from e
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

    with open(params_path, "r", encoding="utf-8") as f:
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
