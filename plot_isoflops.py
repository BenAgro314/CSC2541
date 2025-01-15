
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import json
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import json
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


log_dirs = glob.glob("outputs/*")
flops_to_curve = {}

for log_dir in log_dirs:
    with open(os.path.join(log_dir, "args.json")) as f:
        args = json.load(f)
        seq_len = args["seq_len"]
        batch_size = args["batch_size"]
    json_files = glob.glob(os.path.join(log_dir, "logs/*.json"))
    if len(json_files) == 0:
        continue
    assert len(json_files) == 1, f"{len(json_files)}"
    try:
        with open(json_files[0], "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"warning: failed to read {log_dir}")
        continue
    final_loss = data["Loss/Train_iter_smoothed"][-1]["value"]
    num_iters = data["Loss/Train_iter_smoothed"][-1]["step"]
    params = data["Model/Total Parameters"][-1]["value"]
    num_peta_flops = float(log_dir.split("/")[1].split("_")[0][5:])
    name = log_dir.split("/")[1]
    # outlier rejection based on loss value. Some very short runs do not converge
    if final_loss > 2:
        continue
    if num_peta_flops not in flops_to_curve:
        flops_to_curve[num_peta_flops] = []
    flops_to_curve[num_peta_flops].append((params, final_loss, name))

with open("flops_to_curve.json", "w") as f:
    json.dump(flops_to_curve, f)

min_params = min([v[0] for g in flops_to_curve.values() for v in g])
max_params = max([v[0] for g in flops_to_curve.values() for v in g])
min_flops = min(flops_to_curve.keys())
max_flops = max(flops_to_curve.keys())

cmap = plt.get_cmap("rainbow")

for k,v in flops_to_curve.items():
    flops_to_curve[k] = sorted(flops_to_curve[k], key=lambda x: x[0])
flops_to_curve_items = sorted(flops_to_curve.items(), key=lambda x: x[0])

minima_params = []
minima_flops = []

for num_flops, data in flops_to_curve_items:  # flops_to_curve.items():
    # Skip high flop counts
    print(data)
    if num_flops >= 60:
        continue

    # Clean data if necessary (e.g., remove specific outliers manually)
    new_data = []
    for datum in data:
        name = datum[-1]  # Assuming the last element is a name or identifier
        new_data.append(datum)
    data = new_data

    # Extract x and y values
    xs_log = np.array([math.log10(x[0]) for x in data])  # log10 of x-values
    ys = np.array([x[1] for x in data])                # y-values

    # Reshape for sklearn
    X = xs_log.reshape(-1, 1)
    y = ys

    # Create a pipeline with PolynomialFeatures and RANSACRegressor
    polynomial_degree = 2
    ransac = make_pipeline(
        PolynomialFeatures(degree=polynomial_degree, include_bias=True),
        RANSACRegressor(random_state=42)
    )

    # Fit the model
    ransac.fit(X, y)

    # Access the RANSAC regressor step to get the inlier mask
    ransac_regressor = ransac.named_steps['ransacregressor']
    inlier_mask = ransac_regressor.inlier_mask_
    outlier_mask = ~inlier_mask

    # Separate inliers and outliers
    X_inliers = X[inlier_mask]
    y_inliers = y[inlier_mask]
    X_outliers = X[outlier_mask]
    y_outliers = y[outlier_mask]

    # Extract coefficients from the fitted model
    # Since PolynomialFeatures with include_bias=True, coefficients are [c, b, a]
    poly_features = ransac.named_steps['polynomialfeatures']
    coeffs = ransac_regressor.estimator_.coef_
    intercept = ransac_regressor.estimator_.intercept_

    # Reconstruct polynomial coefficients in descending order for np.poly1d
    # [a, b, c] corresponds to a*x^2 + b*x + c
    a = coeffs[2] if len(coeffs) > 2 else 0
    b = coeffs[1] if len(coeffs) > 1 else 0
    c = intercept

    # Create a polynomial function
    poly = np.poly1d([a, b, c])

    # Generate x values for the fitted curve (smooth curve)
    xs_fit_log = np.linspace(min(xs_log), max(xs_log), 100)
    ys_fit = poly(xs_fit_log)

    # Convert the fitted x values back to original scale for plotting
    xs_fit = 10**xs_fit_log

    # Normalize flop counts for coloring
    log_normalized_flops = (math.log(num_flops) - math.log(min_flops)) / (
        math.log(max_flops) - math.log(min_flops) + 1e-5
    )
    color = cmap(log_normalized_flops)

    # Plot inliers
    plt.scatter(
        10**X_inliers.flatten(),
        y_inliers,
        label=f"pflops={num_flops}",
        color=color,
        marker='o',
        # edgecolor='k',
        alpha=0.7
    )

    # Plot the fitted polynomial curve
    plt.plot(xs_fit, ys_fit, color=color, linewidth=2) # , label=f"Fit pflops={num_flops}")

    # Calculate and store the minima if the quadratic coefficient is non-zero
    if a != 0:
        minima_log = -b / (2 * a)
        minima = 10**minima_log
        y_val = poly(minima_log)

        # Plot minima
        plt.scatter([minima], [y_val], color=color, marker="x", s=100) # , label=f"Minima pflops={num_flops}")

        minima_params.append(minima)
        minima_flops.append(num_flops)
    else:
        print(f"pflops={num_flops}: Quadratic coefficient is zero, cannot compute minima.")


plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=3, fancybox=True, shadow=True)
# plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of Parameters")
plt.ylabel("Training Loss")
plt.savefig("flops_to_curve.png")
plt.close("all")

plt.scatter(minima_flops, minima_params)
# linear best fit
m, b = np.polyfit([math.log10(m) for m in minima_flops], [math.log10(m) for m in minima_params], 1)
plt.plot(minima_flops, [10**(m*math.log10(x) + b) for x in minima_flops], label=f"y={m:.2f}x + {b:.2f}", linestyle="--")
plt.legend()
plt.ylabel("Number of Parameters")
plt.xlabel("Number of Petaflops")
plt.xscale("log")
plt.yscale("log")
plt.savefig("minima_params_vs_flops.png")
plt.close("all")

minima_tokens = [(f*1e15)/(6 * p) for f, p in zip(minima_flops, minima_params)]
plt.scatter(minima_flops, minima_tokens)
m, b = np.polyfit([math.log10(m) for m in minima_flops], [math.log10(m) for m in minima_tokens], 1)
plt.plot(minima_flops, [10**(m*math.log10(x) + b) for x in minima_flops], label=f"y={m:.2f}x + {b:.2f}", linestyle="--")
plt.legend()
plt.ylabel("Number of Tokens")
plt.xlabel("Number of Petaflops")
plt.xscale("log")
plt.yscale("log")
plt.savefig("minima_tokens_vs_flops.png")
plt.close("all")

plt.scatter(minima_params, minima_tokens)
m, b = np.polyfit([math.log10(m) for m in minima_params], [math.log10(m) for m in minima_tokens], 1)
plt.plot(minima_params, [10**(m*math.log10(x) + b) for x in minima_params], label=f"y={m:.2f}x + {b:.2f}", linestyle="--")
plt.legend()
plt.ylabel("Number of Tokens")
plt.xlabel("Number of Params")
plt.xscale("log")
plt.yscale("log")
plt.savefig("minima_tokens_vs_params.png")
plt.close("all")
