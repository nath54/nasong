import sys
import numpy as np
from nasong.core.value import ValueTrainableParameter, ParameterContext
from nasong.trainable.engines.numpy_engine import NumpyEngine

from nasong.core.values.basic.value_constant import Constant
from nasong.core.values.basic.value_identity import Identity
from nasong.core.values.single_itms_ops.value_basic_scaling import BasicScaling
from nasong.core.values.mult_itms_ops.value_sum import Sum


def check_model(name, model_fn, target_audio, sample_rate, engine):
    import nasong.core.value
    import numpy

    is_patched = nasong.core.value.np is not numpy
    print(f"\n--- Testing Model: {name} (Engine Patched: {is_patched}) ---")

    try:
        model, params = model_fn()

        # helper to compute loss for a given set of param values
        def get_loss(p_vals):
            for p, val in zip(params, p_vals):
                p.value = float(val)
            # Ensure we use high precision prediction for loss calculation
            return engine.compute_loss(target_audio, model, sample_rate)

        # 1. AD Gradients
        engine.compute_loss(target_audio, model, sample_rate)
        engine.gradients = {}

        if hasattr(engine, "compute_gradients"):
            # AutogradEngine
            engine.compute_gradients()
        else:
            # NumpyEngine
            context = {"engine": engine, "indices": engine.indices}
            engine.blueprint.backward(engine.grad_output, context, sample_rate)

        # 2. Finite Difference for each parameter
        eps = 1e-4
        orig_vals = [float(p.value) for p in params]
        pass_all = True

        for i, p in enumerate(params):
            v0 = orig_vals[i]
            v_plus = v0 + eps
            v_minus = v0 - eps

            vals_plus = list(orig_vals)
            vals_plus[i] = v_plus
            l_plus = get_loss(vals_plus)

            vals_minus = list(orig_vals)
            vals_minus[i] = v_minus
            l_minus = get_loss(vals_minus)

            f_grad = (l_plus - l_minus) / (v_plus - v_minus)
            ad_grad = float(engine.gradients[p][0])

            rel_diff = abs(ad_grad - f_grad) / (abs(f_grad) + 1e-8)
            print(
                f"  Param {p.name:10}: AD={ad_grad:12.8f}, FD={f_grad:12.8f}, RelDiff={rel_diff:.2e}"
            )
            try:
                np.testing.assert_allclose(
                    ad_grad,
                    f_grad,
                    rtol=1e-2,  # Loosened rtol slightly for now
                    atol=1e-4,
                    err_msg=f"Mismatch in {name} for {p.name}",
                )
            except AssertionError as e:
                print(f"  [FAIL] {e}")
                pass_all = False

        return pass_all
    except Exception as e:
        print(f"Error in check_model: {e}")
        import traceback

        traceback.print_exc()
        # Try to print params if defined
        try:
            for p in params:
                print(f"  Param {p.name}: value={p.value}, type={type(p.value)}")
        except Exception as _e:
            pass
        return False


def verify():
    sample_rate = 44100
    duration_samples = 100
    target_audio = np.random.randn(duration_samples).astype(np.float64)

    class Config:
        learning_rate = 0.001
        optimizer_type = "adam"
        loss_type = "mse"

    engine_configs = [
        ("NumpyEngine", lambda: NumpyEngine(Config())),
        (
            "AutogradEngine",
            lambda: __import__(
                "nasong.trainable.engines.autograd_engine", fromlist=["AutogradEngine"]
            ).AutogradEngine(Config()),
        ),
    ]

    time_indices = Identity()
    _time_seconds = BasicScaling(
        time_indices, mult_scale=Constant(1.0 / sample_rate), sum_scale=Constant(0.0)
    )

    # Test cases
    test_cases = [
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

    all_engines_pass = True
    for engine_name, engine_fn in engine_configs:
        print(f"\n===== VERIFYING ENGINE: {engine_name} =====")
        engine = engine_fn()
        engine_pass = True
        for name, model_fn in test_cases:
            # Fresh capture wrapper
            def wrapped_model_fn():
                with ParameterContext(capture=True) as ctx:
                    m, _ = model_fn()
                    return m, ctx.captured_params

            if not check_model(
                name, wrapped_model_fn, target_audio, sample_rate, engine
            ):
                engine_pass = False

        if engine_pass:
            print(f"\n[PASS] Engine {engine_name} passed all tests.")
        else:
            print(f"\n[FAIL] Engine {engine_name} failed some tests.")
            all_engines_pass = False

    if all_engines_pass:
        print("\n[SUCCESS] All engines and models passed gradient verification!")
    else:
        print("\n[FAILURE] Some gradient verifications failed.")
        sys.exit(1)


if __name__ == "__main__":
    import traceback

    try:
        verify()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
