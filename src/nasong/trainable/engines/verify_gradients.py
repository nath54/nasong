import numpy as np
from nasong.core.value import ValueTrainableParameter, ParameterContext
from nasong.trainable.engines.numpy_engine import NumpyEngine

from nasong.core.values.basic.value_constant import Constant
from nasong.core.values.basic.value_identity import Identity
from nasong.core.values.single_itms_ops.value_basic_scaling import BasicScaling
from nasong.core.values.complex.value_sin import Sin
from nasong.core.values.mult_itms_ops.value_sum import Sum


def check_model(name, model_fn, target_audio, sample_rate, engine):
    print(f"\n--- Testing Model: {name} ---")

    model, params = model_fn()

    # helper to compute loss for a given set of param values
    def get_loss(p_vals):
        for p, val in zip(params, p_vals):
            p.value = val
        return engine.compute_loss(target_audio, model, sample_rate)

    # 1. Manual AD
    engine.compute_loss(target_audio, model, sample_rate)
    engine.gradients = {}
    context = {"engine": engine, "indices": engine.indices}
    engine.blueprint.backward(engine.grad_output, context, sample_rate)

    # 2. Finite Difference for each parameter
    eps = 1e-4
    orig_vals = [float(p.value) for p in params]

    for i, p in enumerate(params):
        v0 = orig_vals[i]
        v_plus = np.float32(v0 + eps)
        v_minus = np.float32(v0 - eps)

        vals_plus = list(orig_vals)
        vals_plus[i] = float(v_plus)
        l_plus = get_loss(vals_plus)

        vals_minus = list(orig_vals)
        vals_minus[i] = float(v_minus)
        l_minus = get_loss(vals_minus)

        f_grad = (l_plus - l_minus) / (float(v_plus) - float(v_minus))
        ad_grad = engine.gradients[p][0]

        rel_diff = abs(ad_grad - f_grad) / (abs(f_grad) + 1e-8)
        print(
            f"  Param {p.name:10}: AD={ad_grad:12.8f}, FD={f_grad:12.8f}, RelDiff={rel_diff:.2e}"
        )
        np.testing.assert_allclose(
            ad_grad,
            f_grad,
            rtol=1e-3,
            atol=1e-5,
            err_msg=f"Mismatch in {name} for {p.name}",
        )


def verify():
    sample_rate = 44100
    duration_samples = 100
    target_audio = np.random.randn(duration_samples).astype(np.float32)

    class Config:
        learning_rate = 0.001
        optimizer_type = "adam"
        loss_type = "mse"

    engine = NumpyEngine(Config())

    time_indices = Identity()
    time_seconds = BasicScaling(
        time_indices, mult_scale=Constant(1.0 / sample_rate), sum_scale=Constant(0.0)
    )

    # Test cases
    test_cases = [
        ("Constant", lambda: (ValueTrainableParameter(1.5, name="amp"), [])),
        (
            "Linear",
            lambda: (
                BasicScaling(
                    time_seconds,
                    mult_scale=ValueTrainableParameter(10.0, name="slope"),
                    sum_scale=ValueTrainableParameter(0.5, name="bias"),
                ),
                [],
            ),
        ),
        (
            "Sin Wave",
            lambda: (
                Sin(
                    time_seconds,
                    frequency=ValueTrainableParameter(10.0 * 2 * np.pi, name="freq"),
                    amplitude=ValueTrainableParameter(0.8, name="amp"),
                ),
                [],
            ),
        ),
        (
            "Sum Model",
            lambda: (
                Sum(
                    [
                        ValueTrainableParameter(0.5, name="v1"),
                        ValueTrainableParameter(0.3, name="v2"),
                    ]
                ),
                [],
            ),
        ),
    ]

    for name, model_fn in test_cases:
        # Fresh capture wrapper
        def wrapped_model_fn():
            with ParameterContext(capture=True) as ctx:
                m, _ = model_fn()
                return m, ctx.captured_params

        check_model(name, wrapped_model_fn, target_audio, sample_rate, engine)

    print("\n[SUCCESS] All models passed gradient verification!")


if __name__ == "__main__":
    verify()
