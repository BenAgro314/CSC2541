
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import json
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import json

READ = False # True

def parse_tensorboard_log(event_file_path, tag):
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

    return scalar_events

log_dirs = glob.glob("outputs/*")

flops_to_curve = {}

if READ:
    for log_dir in log_dirs:
        print(f"Processing {log_dir}")
        with open(os.path.join(log_dir, "args.json")) as f:
            args = json.load(f)
            seq_len = args["seq_len"]
            batch_size = args["batch_size"]
        tensorboard_files = glob.glob(os.path.join(log_dir, "logs/*"))
        assert len(tensorboard_files) == 1
        try:
            params = parse_tensorboard_log(tensorboard_files[0], "Model/Total Parameters")[0].value
            data = parse_tensorboard_log(tensorboard_files[0], "Loss/Train_iter_smoothed")
        except Exception as e:
            continue
        num_iters = data[-1].step
        final_loss = data[-1].value
        #num_flops = num_iters * batch_size * seq_len * params * 6
        num_peta_flops = float(log_dir.split("/")[1].split("_")[0][5:])
        name = log_dir.split("/")[1]

        print(f"params={params}, final_loss={final_loss}, num_pflops={num_peta_flops}")

        # if "d128_l2_h2" in log_dir:
        #     continue

        if num_peta_flops not in flops_to_curve:
            flops_to_curve[num_peta_flops] = []
        flops_to_curve[num_peta_flops].append((params, final_loss, name))
    with open("flops_to_curve.json", "w") as f:
        json.dump(flops_to_curve, f)
else:
    with open("flops_to_curve.json") as f:
        flops_to_curve = json.load(f)
        flops_to_curve = {float(k): v for k, v in flops_to_curve.items()}

min_params = min([v[0] for g in flops_to_curve.values() for v in g])
max_params = max([v[0] for g in flops_to_curve.values() for v in g])
min_flops = min(flops_to_curve.keys())
max_flops = max(flops_to_curve.keys())

cmap = plt.get_cmap("viridis")

for num_flops, data in flops_to_curve.items():
    if num_flops >= 10:
        break
    xs_log = [math.log10(x[0]) for x in data]  # log10 of x-values
    ys = [x[1] for x in data]                  # y-values

    # Fit a parabola (quadratic polynomial) to the data
    # coeffs will contain [a, b, c] for the polynomial a*x^2 + b*x + c
    coeffs = np.polyfit(xs_log, ys, deg=2)
    poly = np.poly1d(coeffs)

    # Generate x values for the fitted curve (smooth curve)
    xs_fit_log = np.linspace(min(xs_log), max(xs_log), 100)
    ys_fit = poly(xs_fit_log)

    # Convert the fitted x values back to original scale for plotting
    xs_fit = [10**x for x in xs_fit_log]

    log_normalized_flops = (math.log(num_flops) - math.log(min_flops)) / (math.log(max_flops) - math.log(min_flops))
    color = cmap(log_normalized_flops)

    # Plot the fitted parabola
    p = plt.scatter([x[0] for x in data], [x[1] for x in data], label=f"pflops={num_flops}", color=color)

    xs_fit_large = np.linspace(math.log10(min_params), math.log10(max_params), 100)
    ys_fit_large = poly(xs_fit_large)
    xs_fit_large = [10**x for x in xs_fit_large]
    plt.plot(xs_fit_large, ys_fit_large, linestyle="--", color=color)
    plt.plot(xs_fit, ys_fit, color=color)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=3, fancybox=True, shadow=True)
# plt.yscale("log")
plt.xscale("log")
plt.savefig("flops_to_curve.png")
plt.xlabel("Number of Parameters")
plt.ylabel("Training Loss")
plt.close("all")