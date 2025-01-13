import json
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import glob
import math

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

if __name__ == "__main__":
    log_dirs = glob.glob("outputs/*")
    params_to_data = {}
    min_params = float('inf')
    max_params = 0
    params_to_data = []
    min_x = 1e-3
    cmap = plt.get_cmap("turbo")
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
    seen_params = set()
    log_params_max = math.log(max_params)
    log_params_min = math.log(min_params)
    for params, data in params_to_data:
        print(f"Plotting {params} params")
        log_params = math.log(params)
        normalized_params = (log_params - log_params_min) / (log_params_max - log_params_min)
        if params not in seen_params:
            plt.plot(data[0], data[1], label=f"{params} params", c=cmap(normalized_params))
        else:
            plt.plot(data[0], data[1], c=cmap(normalized_params))
        seen_params.add(params)
    plt.xlim(left=min_x)
    plt.xlabel("PetaFLOPs")
    plt.ylabel("Training Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("train_per_flops_smoothed.png")

    # Configuration
    # EVENT_FILE_PATH = "outputs/ts_d512_l8_h8_t1e6/logs/events.out.tfevents.1736701636.wb-tor-68CTQP2.1444507.0"
    # TAG = "Loss/Train_per_flops_smoothed"
    # OUTPUT_JSON_PATH = "train_per_flops_smoothed.json"

    # try:
    #     parse_tensorboard_log(EVENT_FILE_PATH, TAG, OUTPUT_JSON_PATH)
    # except Exception as e:
    #     print(f"Error: {e}")
