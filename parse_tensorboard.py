import json
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import glob
import math
import numpy as np

def parse_tensorboard_log(event_file_path, tag, step_multiplier=1):
    """
    Parses a TensorBoard event file to extract scalar data for a given tag and saves it as JSON.

    Parameters:
    - event_file_path (str): Path to the TensorBoard event file.
    - tag (str): The tag name of the scalar to extract.
    - output_json_path (str): Path to save the output JSON file.
    """
    if not os.path.isfile(event_file_path):
        raise FileNotFoundError(f"Event file not found: {event_file_path}")

    # Initialize EventAccumulator with scalar data
    ea = EventAccumulator(event_file_path, size_guidance={'scalars': 0})
    ea.Reload()

    # Check if the tag exists
    if tag not in ea.Tags()['scalars']:
        available_tags = ea.Tags()['scalars']
        raise ValueError(f"Tag '{tag}' not found in event file. Available tags: {available_tags}")

    # Retrieve scalar events for the specified tag
    scalar_events = ea.Scalars(tag)

    # Prepare data for JSON
    x = []
    y = []
    for event in scalar_events:
        x.append(event.step * step_multiplier)
        y.append(event.value)

    # Save to JSON
    # with open(output_json_path, 'w') as json_file:
        # json.dump(data, json_file, indent=4)

    # print(f"Successfully saved {len(data)} data points to {output_json_path}")
    return x, y

READ_BOARD = False
if __name__ == "__main__":
    log_dirs = glob.glob("outputs/*")
    min_params = float('inf')
    max_params = 0
    params_to_data = []
    min_x = 1e-3
    cmap = plt.get_cmap("viridis")
    if READ_BOARD:
        for log_dir in log_dirs:
            print(f"Processing {log_dir}")
            with open(os.path.join(log_dir, "args.json")) as f:
                args = json.load(f)
                seq_len = args["seq_len"]
                batch_size = args["batch_size"]
            tensorboard_files = glob.glob(os.path.join(log_dir, "logs/*"))
            assert len(tensorboard_files) == 1
            params = parse_tensorboard_log(tensorboard_files[0], "Model/Total Parameters")[1][0]
            step_multiplier = batch_size * 6 * params * seq_len / 1e15  # 1e15 corresponds to petaflops
            data = parse_tensorboard_log(tensorboard_files[0], "Loss/Train_iter_smoothed", step_multiplier)
            # params_to_data[params] = data
            params_to_data.append((params, data))
            min_params = min(min_params, params)
            max_params = max(max_params, params)
        with open("train_per_flops_smoothed.json", "w") as f:
            json.dump(params_to_data, f)
    else:
        with open("train_per_flops_smoothed.json") as f:
            params_to_data = json.load(f)

    # Existing snippet to plot each curve using a colormap
    seen_params = set()
    params_to_data = sorted(params_to_data, key=lambda x: (x[0], len(x[1])))
    min_params = params_to_data[0][0]
    max_params = params_to_data[-1][0]
    log_params_max = math.log(max_params)
    log_params_min = math.log(min_params)
    warmup = 1000
    x_min = None
    x_max = 0

    batch_size = 128

    # Plot all curves
    for params, data in params_to_data:
        print(f"Plotting {params} params")
        log_params = math.log(params)
        normalized_params = (log_params - log_params_min) / (log_params_max - log_params_min)
        # Skip the initial `warmup` points

        flops_arr = data[0][warmup:]
        loss_arr = data[1][warmup:]

        if params not in seen_params:
            plt.plot(flops_arr, loss_arr, label=f"{params} params", c=cmap(normalized_params))
        else:
            plt.plot(flops_arr, loss_arr, c=cmap(normalized_params))
        seen_params.add(params)
        if x_min is None:
            x_min = flops_arr[-1]
        x_max = max(flops_arr[-1], x_max)


    # ------------------------------------------------------------------
    # 1) Build a global list of all FLOPs (after warmup) to find min/max
    # ------------------------------------------------------------------
    # all_flops = []
    # for _, data in params_to_data:
    #     flops_arr = data[0][warmup:]
    #     all_flops.extend(flops_arr)

    # # global_min_flops = min(all_flops)
    # global_max_flops = max(all_flops)

    # ------------------------------------------------------------------
    # 2) Create 1500 log-spaced FLOPs from global_min_flops to global_max_flops
    # ------------------------------------------------------------------
    num_points = 1500
    flops_grid = np.logspace(
        math.log10(x_min),
        math.log10(x_max),
        num=num_points
    )

    # ------------------------------------------------------------------
    # 3) For each flop in flops_grid, interpolate each curve’s loss
    #    and take the minimum
    # ------------------------------------------------------------------
    min_loss_envelope = []
    best_params = []
    best_tokens = []

    for flop_val in flops_grid:
        # Track the best (minimum) loss among all curves at this flop value
        best_loss = float('inf')
        best_param_count = None
        best_token_count = None
        for params, data in params_to_data:
            flops_arr = data[0][warmup:]
            loss_arr = data[1][warmup:]
            # Interpolate training loss at flop_val
            # np.interp returns the value from the piecewise linear interpolation
            # If flop_val is outside the range, it uses the edge values by default.
            y_val = np.interp(flop_val, flops_arr, loss_arr, left=float('inf'))
            if y_val < best_loss:
                best_loss = y_val
                best_param_count = params
                best_token_count = flop_val / (batch_size * 6 * params)
        min_loss_envelope.append(best_loss)
        best_params.append(best_param_count)
        best_tokens.append(best_token_count)

    # ------------------------------------------------------------------
    # 4) Plot the envelope on top (in black or any color you prefer)
    # ------------------------------------------------------------------
    plt.plot(flops_grid, min_loss_envelope, label="Min-Loss Envelope", 
            color="black", linewidth=2)

    # ------------------------------------------------------------------
    # 5) Finalize and Save
    # ------------------------------------------------------------------
    # Optionally shift left x-limit to see enough data
    plt.xlim(left=flops_grid[0] * 0.8)

    plt.xlabel("PetaFLOPs")
    plt.ylabel("Training Loss")
    plt.xscale("log")

    # Set y-axis to log scale (base 2)
    plt.yscale('log', base=2)

    # Manually set y-ticks (positions) and labels
    y_ticks = [0.5, 1, 2, 3]
    plt.yticks(y_ticks, [str(y) for y in y_ticks])  # Display values as labels

    plt.legend()
    plt.savefig("train_per_flops_smoothed.png")

    # now plot the minium parameter valeu
    plt.close("all")
    plt.scatter(flops_grid, best_params)
    plt.xlabel("PetaFLOPs")
    plt.ylabel("Parameters")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("params_per_flops.png")

    plt.close("all")
    plt.scatter(flops_grid, best_tokens)
    plt.xlabel("PetaFLOPs")
    plt.ylabel("Tokens")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("tokens_per_flops.png")