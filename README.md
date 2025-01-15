# Minchilla: A minima reproduction of the Chinchilla Scaling Laws

In this repo we apply method 2 (iso-flop curves) from the Chinchilla Scaling Laws Paper
(["Training Compute-Optimal Large Language Models" by Hoffmann et a.](https://arxiv.org/pdf/2203.15556))
to **very small transformers** on a **character-level lanugage modelling task**.

In this setting, we find a similar result that parameters and training tokens should be scaled up in
roughly equal proportions.

$$N_{opt} \propto C^{0.48}, \quad D_{opt} \propto C^{0.52}$$
where $D$ is the number of training tokens, $N$ is the number of model parameters, and $C$ is the number of FLOPs available for training.

![IsoFlop curve](resources/isoflop_curve.png)


## Environemnt setup

```bash
conda env create -f environment.yml
conda deactivate
conda activate minchilla
```

## Run trainings

Change `NUM_CUDA_DEVICES` and `MAX_CONCURRENT_PROCESSES` in `run_isoflops.py`.
Then run `python run_isoflops.py`

These trainings may take a while, depending on your machine. It took around 12 hours for us on 2 A5000's.
Alternatively, we have pre-saved the relevant results to `resources/flops_to_curve.json`.
We also saved raw training results in `saved_outputs/` if you are interested. (TODO)

## Estiamte scaling laws

Run `plot_isoflops.py True`, which will produce the visualizations in `resources/`
If you didn't run the above training and instead want to re-use the results in `flops_to_curve.json`,
then run `plot_isoflops.py False`

![Tokens vs flops](resources/minima_tokens_vs_flops.png)
![Params vs flops](resources/minima_params_vs_flops.png)
![Tokens vs params](resources/minima_tokens_vs_params.png)

