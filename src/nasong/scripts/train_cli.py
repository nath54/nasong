import os
import argparse
import time
import json
import torch
import torch.optim as optim
import nasong.core.value as lv
import nasong.trainable.extract as learnable
from nasong.trainable.train import (
    load_wav_segment,
    spectral_loss,
    extract_note_parameters,
)
from nasong.scripts.experiment_manager import ExperimentManager


def train_instrument(
    instrument_name: str,
    target_wav: str,
    epochs: int = 1000,
    learning_rate: float = 0.05,
    output_dir: str = None,
    device_name: str = "cpu",
):
    manager = ExperimentManager()

    # Create experiment
    params = {
        "instrument": instrument_name,
        "target_wav": target_wav,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": device_name,
    }
    exp = manager.create_experiment(name=f"train_{instrument_name}", params=params)
    print(f"Started experiment: {exp.id} at {exp.path}")

    # Load target
    target_audio, sample_rate = load_wav_segment(target_wav, duration=2.0)
    target_tensor = torch.tensor(target_audio, device=device_name)

    # Extract notes to guide training (optional, for start_time/duration)
    # For now, simplistic single note assumption or use extracted params
    # We'll use a fixed note for simplicity of this CLI MVP
    note_start = 0.1
    note_duration = 1.0

    # Initialize instrument with Capture Context to find parameters
    print("Initializing instrument and capturing parameters...")

    # We need a dummy time and frequency to instantiate the graph
    # But wait! ValueTrainableParameter are created inside the function.
    # We need to run the function ONCE to create the graph and capture parameters.

    time_val = lv.Value()  # Dummy base
    # In a real scenario, we'd use a Variable or Ramp for time, but here we just need to trace
    # Actually, nasong design generates the graph. We need to instantiate the graph.

    # Let's create the graph inputs
    # Time is usually a Ramp in the render loop, but for graph construction it's just a Value
    # We need to handle the fact that 'time' argument in instrument is a Value.

    # In training loop:
    # 1. Instantiate instrument (creates parameters)
    # 2. Render audio
    # 3. Loss
    # 4. Backward

    # To capture parameters, we use the Context
    captured_params = []

    with lv.ParameterContext(capture=True) as ctx:
        # Create inputs
        # We need actual values that can be used in the graph
        # For training, 'time' is usually a generic Value, but during render it's a Ramp.
        # The instrument function builds the graph.

        # We use simple placeholders for graph construction
        t_placeholder = lv.Ramp(0, 1, sample_rate)  # Just to have a valid Value
        f_placeholder = lv.Constant(440.0)

        # Instantiate
        blueprint = learnable.get_trainable_instrument(instrument_name)
        instrument_graph = blueprint(
            t_placeholder, f_placeholder, note_start, note_duration
        )

        captured_params = ctx.captured_params

    print(f"Captured {len(captured_params)} trainable parameters.")

    if not captured_params:
        print(
            "No parameters to train! specific instrument might not use ValueTrainableParameter."
        )
        return

    # Optimizer
    # We need the .value attribute from each parameter
    param_tensors = [p.value for p in captured_params]
    # Ensure they require grad
    for p in param_tensors:
        p.requires_grad = True

    optimizer = optim.Adam(param_tensors, lr=learning_rate)

    # Training Loop
    best_loss = float("inf")
    start_time = time.time()

    # Render buffer (time)
    duration = 2.0
    num_samples = int(duration * sample_rate)

    # We need a proper render function that supports Torch
    # nasong.core.value doesn't have a specific "render_graph" utility that takes a root Value
    # We usually call getitem_torch on the root.

    # Pre-compute time index tensor
    time_indices = torch.arange(num_samples, device=device_name, dtype=torch.float32)

    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Render
            # We call getitem_torch on the root of the graph
            synthesized = instrument_graph.getitem_torch(
                time_indices, sample_rate, device=device_name
            )

            # Pad/Crop to match target
            if synthesized.shape[0] > target_tensor.shape[0]:
                synthesized = synthesized[: target_tensor.shape[0]]
            elif synthesized.shape[0] < target_tensor.shape[0]:
                # Warning: zero padding might affect loss?
                synthesized = torch.nn.functional.pad(
                    synthesized, (0, target_tensor.shape[0] - synthesized.shape[0])
                )

            # Loss
            loss = spectral_loss(synthesized, target_tensor, sample_rate)

            loss.backward()
            optimizer.step()

            curr_loss = loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {curr_loss:.4f}")

            if curr_loss < best_loss:
                best_loss = curr_loss
                # Checkpoint parameters?
                # We can save them to the experiment

                # Extract values map
                # Since we don't have names yet (unless user added them), we use index
                current_params_values = {}
                for i, p in enumerate(captured_params):
                    name = p.name if p.name else f"param_{i}"
                    current_params_values[name] = float(p.value.item())

                exp.save_parameters_json(current_params_values)
                exp.status = "training"
                exp.metrics["best_loss"] = best_loss
                exp.metrics["current_epoch"] = epoch
                exp.save_meta()

    except KeyboardInterrupt:
        print("Training interrupted.")
        exp.status = "interrupted"
    except Exception as e:
        print(f"Error: {e}")
        exp.status = "failed"
        exp.save_meta()
        raise e
    else:
        exp.status = "completed"

    exp.metrics["duration"] = time.time() - start_time
    exp.save_meta()
    print(f"Training finished. Best Loss: {best_loss}")
    print(f"Parameters saved to {exp.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Nasong instrument")
    parser.add_argument("--instrument", required=True, help="Instrument name")
    parser.add_argument("--target", required=True, help="Target WAV file")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    train_instrument(
        args.instrument, args.target, args.epochs, args.lr, device_name=args.device
    )
