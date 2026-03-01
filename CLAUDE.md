# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install for development:**
```bash
pip install -e .[test]         # core + test deps (pytest, diffusers)
pip install -e .[examples]     # adds jupyter, matplotlib, transformers, etc.
```

**Run tests:**
```bash
python -m pytest               # fast tests only (default)
python -m pytest --run_slow    # include slow tests (require GPU/downloads)
python -m pytest tests/test_diffusion.py::TestSchedule::test_DDPMScheduler  # single test
```

**Run example training scripts (requires `accelerate config` first):**
```bash
accelerate launch examples/fashion_mnist_dit.py
accelerate launch examples/fashion_mnist_unet.py
accelerate launch examples/cifar_unet.py
```

**Build and publish:**
```bash
pip install -e .[dev]
python -m build
twine upload dist/*
```

## Architecture

The library is in `src/smalldiffusion/` with five modules:

### Core: `diffusion.py`
Contains the entire diffusion training/sampling logic (~143 lines). Three key abstractions:

- **`Schedule`** — wraps a tensor of increasing sigma values. Subclasses: `ScheduleLogLinear`, `ScheduleDDPM`, `ScheduleLDM`, `ScheduleSigmoid`, `ScheduleCosine`. Key methods: `sample_sigmas(steps)` for inference (trailing spacing), `sample_batch(x0)` for training.
- **`training_loop(loader, model, schedule, ...)`** — a generator that yields a `SimpleNamespace` of locals each iteration (including `loss`). Uses `accelerate` for multi-GPU.
- **`samples(model, sigmas, gam, mu, ...)`** — a generator that yields intermediate `xt` at each denoising step. Generalizes DDPM (`gam=1, mu=0.5`), DDIM (`gam=1, mu=0`), and the accelerated sampler (`gam=2, mu=0`).

### Models: `model.py`
- **`ModelMixin`** — base mixin all models inherit. Provides `rand_input(batchsize)`, `get_loss()`, `predict_eps()`, and `predict_eps_cfg()` (classifier-free guidance). Models must set `self.input_dims`.
- **Modifier decorators** — `Scaled(cls)`, `PredX0(cls)`, `PredV(cls)` are class-factory functions that wrap a model class to change its input scaling or prediction target. Usage: `Scaled(PredX0(DiT))(...)`.
- **`TimeInputMLP`** — simple MLP for 2D toy data; concatenates sinusoidal sigma embeddings with input.
- **`ConditionalMLP`** — extends `TimeInputMLP` with label conditioning via `CondEmbedderLabel`.
- **`SigmaEmbedderSinCos`**, **`CondEmbedderLabel`** — reusable embedding modules for larger models.

### Models: `model_dit.py` and `model_unet.py`
- **`DiT`** — Diffusion Transformer (Peebles & Xie 2022). Accepts optional `cond_embed` kwarg for conditional generation.
- **`Unet`** — U-Net for pixel-space image diffusion.

Both inherit `ModelMixin` and follow the same calling convention: `model(x, sigma, cond=None)`.

### Data: `data.py`
- **`MappedDataset`** — wraps any Dataset with a transform fn; used to strip labels before passing to `training_loop`.
- Toy datasets: `Swissroll`, `DatasaurusDozen`, `TreeDataset` (conditional, returns `(point, leaf_label)` tuples).
- Image helpers: `img_train_transform`, `img_normalize`.

## Key Conventions

**Model calling convention:** All models are called as `model(x, sigma, cond=None)` and return predicted noise (eps) with same shape as `x`. `sigma` can be a scalar tensor `[]` or batched `[B, 1, ..., 1]`.

**Conditional training:** Pass `conditional=True` to `training_loop`. The DataLoader must yield `(data, labels)` tuples. `TreeDataset` does this natively; for standard datasets, use `MappedDataset` or construct a custom Dataset.

**Slow tests:** Tests marked `@pytest.mark.run_slow` are skipped by default. They test full training pipelines for image models and require downloading datasets.

**sigma vs timestep:** The library works entirely in sigma-space (not discrete timestep-space). Schedules store a tensor of sigmas; the `ScheduleDDPM`/`ScheduleLDM` constructors convert from beta parameterization via `sigmas_from_betas`.
