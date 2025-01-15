
from tqdm import tqdm
from char_scaling_laws import SmallTransformer
import math
import subprocess
import sys
import threading
import os

NUM_CUDA_DEVICES = 2
MAX_CONCURRENT_PROCESSES = 2

def count_params(d_model: int, n_heads: int, n_layers: int):
    model = SmallTransformer(vocab_size=174, d_model=d_model, n_heads=n_heads, num_layers=n_layers)
    return sum(p.numel() for p in model.parameters())

def flops_to_params_and_tokens(flops: int, param_count):
    num_tokens = flops / (6 * param_count)
    if num_tokens > 1_900_000_000: # too many tokens, this is the size of our dataset
        return None
    return num_tokens

semaphore = threading.Semaphore(MAX_CONCURRENT_PROCESSES)

def run_subprocess(command, env):
    with semaphore:
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"Error running command: {' '.join(command)}", file=sys.stderr)
                print(f"stderr: {stderr.decode()}", file=sys.stderr)
            else:
                print(f"Successfully executed: {' '.join(command)}")
        except Exception as e:
            print(f"Exception occurred while running command: {' '.join(command)}", file=sys.stderr)
            print(str(e), file=sys.stderr)

flop_counts = [
    1e15, # 1 PetaFLOP
    3e15, 
    6e15, 
    1e16, 
    3e16,  # 30 PetaFLOPs
    # 6e16, 
    # 1e17, 
    # 3e17, # 30 PetaFLOPs
    # 6e17, # 60 PetaFLOPs
]


model_sizes = [
    (d, max(1, d//64), max(d//64, 2)) for d in range(256, 64 * 17, 64)
]
print("Model sizes:")
import matplotlib.pyplot as plt
xs = []
for model_size in model_sizes:
    d_model, n_heads, n_layers = model_size
    print(f"d_model={d_model}, n_heads={n_heads}, num_layers={n_layers}")
    params = count_params(d_model, n_heads, n_layers)
    xs.append(params)
plt.scatter(xs, [1 for _ in range(len(xs))])
plt.xscale("log")
plt.savefig("params.png")
plt.close("all")

count = 0
batch_size = 128
seq_len = 128
for flop_count in flop_counts:
    print("=" * 30)
    for d_model, n_heads, n_layers in model_sizes:
        params = count_params(d_model, n_heads, n_layers)
        tokens = flops_to_params_and_tokens(flop_count, params)
        
        if tokens is None:
            print(f"Skipping (tokens={tokens}) for flop_count={flop_count}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
            continue
        train_iters = tokens / (batch_size * seq_len)
        print(f"train_iters={train_iters}")
        if train_iters < 100:
            print(f"Skipping (train_iters={train_iters}) for flop_count={flop_count}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
            continue
        
        s = f"\tTokens: {tokens:.2e}"
        petaflops = flop_count / 1e15
        print(f"petaflops={petaflops} | params={params} (d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}) | tokens={s}")

        cuda_device = count % NUM_CUDA_DEVICES  # Adjust based on available CUDA devices

        # Prepare the environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        command = [
            sys.executable,  # Ensures the same Python interpreter is used
            "char_scaling_laws.py",
            "--experiment_name", f"flops{petaflops}_d{d_model}_l{n_layers}_h{n_heads}_tokens{int(tokens)}_params{params}",
            "--num_train_tokens", str(int(tokens)),
            "--n_layers", str(n_layers),
            "--d_model", str(d_model),
            "--n_heads", str(n_heads),
            "--batch_size", str(batch_size),
            "--seq_len", str(seq_len),
        ]
        print("Executing command:", " ".join(command))
        
        threading.Thread(target=run_subprocess, args=(command, env)).start()
        
        count += 1