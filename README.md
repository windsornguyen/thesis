# Spectral State Space Language Model
Repository containing minimal code to pretrain and run a spectral state space language model.


## Installation
Supports Python >= 3.9, < 3.12 (PyTorch Dynamo is not yet supported on Python 3.12+).

### Required packages
It is recommended to use a dependency manager such as [Poetry](https://python-poetry.org/) or [uv](https://github.com/astral-sh/uv) to install the required packages.

First, you should create a virtual environment with
```
python -m venv .venv
source .venv/bin/activate
```

### Installation with uv
Install the required packages with:

```
uv pip install -r requirements.txt
```

If you don't have the uv dependency manager, you can install it with

```
pip install uv
```

or you can run

```
pip install -r requirements.txt
```
instead.

### Installation with Poetry
Run the following commands in the root directory:
```
uv pip install poetry
poetry config virtualenvs.in-project true
poetry shell
poetry install --no-root
```

## Inference
Place the model checkpoint in the `inference` directory. Then, run this command in the root directory:
```
python -m inference.inference
```

## References

[1]: [Spectral State Space Models](https://arxiv.org/abs/2312.06837), Agarwal et al. 2024
