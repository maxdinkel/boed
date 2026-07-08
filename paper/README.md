## Reproducing the Paper Experiments

To install the additional dependencies required for the examples presented in the paper:

```bash
pip install .[paper]
```

---

#### Optional GPU Support

To compute the reference EIG in the pharmacokinetic model and lung model example we use JAX. 
If you have an NVIDIA GPU, you can install a CUDA-enabled JAX backend.

```bash
pip install .[paper,cuda12]
```
or
```bash
pip install .[paper,cuda13]
```

#### Reproducibility

The experiments and figures reported in the paper were generated with the following package versions:

- Python 3.12.13
- NumPy 2.4.6
- SciPy 1.18.0
- JAX 0.10.2 with the CUDA 12 backend
- Matplotlib 3.10.9
- Numba 0.66.0
- tqdm 4.68.3

## Running the experiments
### Synthetic Test Case with Analytical Solution
```bash
python -m paper.synthetic.run
python -m paper.synthetic.plot
```

### Calibration of a Pharmacokinetic Model
```bash
python -m paper.pharmacokinetic.run
python -m paper.pharmacokinetic.plot
python -m paper.pharmacokinetic.compute_reference
```

### Calibration of a Human Lung Model
```bash
python -m paper.lung.run
python -m paper.lung.plot
python -m paper.lung.compute_reference
```

### Calibration of a Viscoplastic Material Model
```bash
python -m paper.viscoplastic.run
python -m paper.viscoplastic.plot
python -m paper.viscoplastic.compute_reference
```
