import sys
from contextlib import contextmanager
import numpy as np
from numpy.typing import NDArray
import autograd.numpy as anp
from autograd import grad
from typing import Dict, Any, List

import nasong.core.all_values
from nasong.core.value import Value, ValueTrainableParameter
from nasong.trainable.engines.base import BaseTrainingEngine


class AutogradEngine(BaseTrainingEngine):
    """
    Training engine using the 'autograd' library for automatic differentiation.

    It works by patching the 'np' attribute in all Value modules to use
    autograd.numpy instead of the standard numpy.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.learning_rate = getattr(config, "learning_rate", 0.001)
        self.optimizer_type = getattr(config, "optimizer_type", "adam")
        self.loss_type = getattr(config, "loss_type", "mse")

        # State for Adam
        self.m: Dict[ValueTrainableParameter, Any] = {}
        self.v: Dict[ValueTrainableParameter, Any] = {}
        self.t = 0

        self.captured_params: List[ValueTrainableParameter] = []
        self.gradients: Dict[ValueTrainableParameter, NDArray[np.float64]] = {}

    @contextmanager
    def _patch_context(self):
        """Temporary replacement of standard numpy with autograd.numpy."""

        # Modules to patch (under nasong.core.values or the base Value class)
        to_patch_names = [
            m
            for m in sys.modules
            if (
                "nasong.core.values" in m
                or "nasong.core.value" in m
                or "nasong.core" in m
            )
            and "autograd_engine" not in m
        ]
        print(f"DEBUG: to_patch_names={to_patch_names}")

        orig_nps = {}
        for module_name in to_patch_names:
            module = sys.modules[module_name]
            if hasattr(module, "np"):
                # Save only once
                if module_name not in orig_nps:
                    orig_nps[module_name] = module.np
                setattr(module, "np", anp)
        try:
            yield
        finally:
            for module_name, orig_np in orig_nps.items():
                if module_name in sys.modules:
                    setattr(sys.modules[module_name], "np", orig_np)

    def _collect_parameters(
        self, node: Any, seen: set
    ) -> List[ValueTrainableParameter]:
        """Deeply traverses the Value graph to find all trainable parameters."""
        params = []
        if id(node) in seen:
            return params
        seen.add(id(node))

        if isinstance(node, ValueTrainableParameter):
            params.append(node)

        # Search for children (anything that is a Value or list/dict of Values)
        if hasattr(node, "__dict__"):
            for val in node.__dict__.values():
                if isinstance(val, Value):
                    params.extend(self._collect_parameters(val, seen))
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if isinstance(item, Value):
                            params.extend(self._collect_parameters(item, seen))
                elif isinstance(val, dict):
                    for item in val.values():
                        if isinstance(item, Value):
                            params.extend(self._collect_parameters(item, seen))

        return params

    def compute_loss(
        self, target_audio: Any, blueprint: Value, sample_rate: int
    ) -> float:
        # Collect parameters
        self.captured_params = self._collect_parameters(blueprint, set())

        # Store as autograd-compatible arrays
        self.target_audio = anp.array(target_audio, dtype=anp.float64)
        self.blueprint = blueprint
        self.sample_rate = sample_rate
        self.indices = anp.arange(len(target_audio), dtype=anp.float64)

        with self._patch_context():
            prediction = blueprint.getitem_np(self.indices, sample_rate)
            loss = anp.mean(anp.square(prediction - self.target_audio))

        self.current_loss = float(loss)
        return self.current_loss

    def compute_gradients(self):
        """Computes gradients using autograd."""
        if not self.captured_params:
            return

        def loss_wrapper(param_values):
            # Inject values
            for i, p in enumerate(self.captured_params):
                p.value = param_values[i]

            prediction = self.blueprint.getitem_np(self.indices, self.sample_rate)
            print(
                f"DEBUG: prediction type={type(prediction)}, shape={getattr(prediction, 'shape', 'N/A')}"
            )
            print(
                f"DEBUG: target type={type(self.target_audio)}, shape={self.target_audio.shape}"
            )
            diff = prediction - self.target_audio
            print(f"DEBUG: diff type={type(diff)}, shape={diff.shape}")
            return anp.mean(anp.square(diff))

        with self._patch_context():
            grad_fn = grad(loss_wrapper)
            # Use anp.array of floats for the input
            p_vals = anp.array(
                [float(p.value) for p in self.captured_params], dtype=anp.float64
            )
            grads = grad_fn(p_vals)

        # Store gradients
        self.gradients = {}
        for i, p in enumerate(self.captured_params):
            self.gradients[p] = np.array([float(grads[i])], dtype=np.float64)
            # Restore to float
            val = p.value
            if hasattr(val, "_value"):
                val = val._value
            p.value = float(val)

    def step(self) -> Dict[str, float]:
        self.compute_gradients()

        for p in self.captured_params:
            if p not in self.gradients:
                continue

            g = self.gradients[p][0]

            if self.optimizer_type == "sgd":
                p.value = float(p.value) - self.learning_rate * g
            elif self.optimizer_type == "adam":
                if p not in self.m:
                    self.m[p] = 0.0
                    self.v[p] = 0.0

                self.t += 1
                beta1, beta2 = 0.9, 0.999
                eps = 1e-8

                self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
                self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g**2)

                m_hat = self.m[p] / (1 - beta1**self.t)
                v_hat = self.v[p] / (1 - beta2**self.t)

                p.value = float(p.value) - self.learning_rate * m_hat / (
                    np.sqrt(v_hat) + eps
                )

            # Ensure float
            val = p.value
            if hasattr(val, "_value"):
                val = val._value
            p.value = float(val)

        return {"loss": self.current_loss, "lr": self.learning_rate}

    def get_parameter_values(self) -> Dict[str, float]:
        return {p.name or str(id(p)): float(p.value) for p in self.captured_params}

    def set_parameter_values(self, parameters: Dict[str, float]) -> None:
        for p in self.captured_params:
            if p.name and p.name in parameters:
                p.value = parameters[p.name]
