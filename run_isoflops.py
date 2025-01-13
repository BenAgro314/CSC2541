
from tqdm import tqdm
from char_scaling_laws import SmallTransformer
import math


def flops_to_params_and_tokens(flops: int, min_params=100_000, max_params=20_000_000, num_points: int = 10):
    log_min = math.log10(min_params)
    log_max = math.log10(max_params) 
    step = (log_max - log_min) / (num_points - 1)
    param_counts = [10**(log_min + i * step) for i in range(num_points)]

    res = []
    for param_count in param_counts:
        num_tokens = flops / (6 * param_count)
        if num_tokens > 1_900_000_000: # too many tokens
            continue
        res.append((param_count, num_tokens))
    return res


flop_counts = [
    1e14, 
    3e14, 
    6e14, 
    1e15, # 1 PFLOP
    3e15, 
    6e15, 
    1e16, 
    3e16, 
]

for flop_count in flop_counts:
    print(f"PFLOP: {flop_count / 1e15}")
    params_and_tokens = flops_to_params_and_tokens(flop_count, num_points=10)
    for param_count, num_tokens in params_and_tokens:
        training_steps = num_tokens / (128 * 128)
        print(f"\tParams: {param_count:.2e}, Tokens: {num_tokens:.2e}, Training Steps: {training_steps}")
    print("=" * 20)
    