
from tqdm import tqdm
from char_scaling_laws import SmallTransformer
import math
import subprocess
import sys
import threading
import os

def count_params(d_model: int, n_heads: int, n_layers: int):
    model = SmallTransformer(vocab_size=174, d_model=d_model, n_heads=n_heads, num_layers=n_layers)
    return sum(p.numel() for p in model.parameters())

def flops_to_params_and_tokens(flops: int, param_count):
    num_tokens = flops / (6 * param_count)
    if num_tokens > 1_900_000_000: # too many tokens
        return None
    return num_tokens

MAX_CONCURRENT_PROCESSES = 4
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
                # Optionally, handle stdout if needed
                # print(stdout.decode())
        except Exception as e:
            print(f"Exception occurred while running command: {' '.join(command)}", file=sys.stderr)
            print(str(e), file=sys.stderr)

flop_counts = [
    # 1e14, 
    # 3e14, 
    # 6e14, 
    1e15, # 1 PFLOP
    3e15, 
    # 6e15, 
    # 1e16, 
    # 3e16, 
]


# d_model, n_heads, n_layers
model_sizes = [ 
    (64, 1, 1),
    (64, 2, 2),
    (64, 2, 3),
    #
    (128, 2, 2),
    (128, 2, 3),
    (128, 2, 4),
    #
    (256, 4, 4),
    (256, 4, 5),
    (256, 4, 6),
    #
    (512, 8, 8),
    (512, 8, 9),
    (512, 8, 10),
]

count = 0
for flop_count in flop_counts:
    print("=" * 30)
    for d_model, n_heads, n_layers in model_sizes:
        params = count_params(d_model, n_heads, n_layers)
        tokens = flops_to_params_and_tokens(flop_count, params)
        
        if tokens is None:
            print(f"Skipping (tokens={tokens}) for flop_count={flop_count}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
            continue
        
        s = f"\tTokens: {tokens:.2e}"
        petaflops = flop_count / 1e15
        print(f"petaflops={petaflops} | params={params} (d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}) | tokens={s}")

        cuda_device = count % 4  # Adjust based on available CUDA devices

        # Prepare the environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        # Construct the command as a list
        command = [
            sys.executable,  # Ensures the same Python interpreter is used
            "char_scaling_laws.py",
            "--experiment_name", f"flops{petaflops}_d{d_model}_l{n_layers}_h{n_heads}_t{int(tokens)}",
            "--num_train_tokens", str(int(tokens)),
            "--n_layers", str(n_layers),
            "--d_model", str(d_model),
            "--n_heads", str(n_heads)
        ]
        print("Executing command:", " ".join(command))
        
        # Launch the subprocess in a separate thread to avoid blocking
        threading.Thread(target=run_subprocess, args=(command, env)).start()
        
        count += 1