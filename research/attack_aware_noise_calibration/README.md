# Code for "Attack-Aware Noise Calibration for Differential Privacy"

This is the accompanying code to the paper "[Attack-Aware Noise Calibration for Differential
Privacy](https://arxiv.org/abs/2407.02191)".

Install the environment with:
```
uv sync
```

To reproduce the plots in the paper, run:
```
uv run jupytext --to ipynb research/attack_aware_noise_calibration/notebooks/*.py
uv run jupyter notebook
```

Then, run each notebook in Jupyter.

We provide information from experimental runs from DP-SGD in the
experiments/data folder. To reproduce these, run:

* https://github.com/ftramer/Handcrafted-DP/blob/main/scripts/run_cnns_cifar10.py
* https://github.com/microsoft/dp-transformers/blob/main/research/fine_tune_llm_w_qlora/fine-tune-nodp.py

with the parameters mentioned in the appendix of the paper. Note that the scripts above require
minor modifications to work with the parameters (i.e., adjust LoRA layers for GPT-2; by default they are set
for Mistral), and additional instrumentation to output the test accuracy metric and the mechanism parameters.
