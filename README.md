# Minchilla: A minima reproduction of the Chinchilla Scaling Laws

## Environemnt setup

```bash
conda env create -f environment.yml
conda deactivate
conda activate minchilla
```

## Run trainings

Change `NUM_CUDA_DEVICES` and `MAX_CONCURRENT_PROCESSES` in `run_isoflops.py`.
Then run `python run_isoflops.py`

These trainings will take a while, depending on your machine.
We have pre-saved the training outputs results

