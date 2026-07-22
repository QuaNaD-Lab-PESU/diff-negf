# diff-negf

**A benchmark-validated differentiable effective-mass NEGF solver for
gradient-based inverse design of quantum transport, with an open
potential-to-transmission dataset.**

QuaNaD Laboratory, Department of ECE, PES University (EC Campus), Bengaluru.

This repository accompanies the paper *"A Benchmark-Validated Differentiable
Effective-Mass NEGF Solver for Gradient-Based Inverse Design of Quantum
Transport, with an Open Potential-to-Transmission Dataset"* and contains the
complete stack: solver, differentiable implementation, verification scripts,
the dataset with designated out-of-family (OOD) splits, and scripts that
regenerate every figure and table in the paper.

## Layout

```
src/               core solvers
  negf_numpy.py      classical 1D coherent-NEGF (reference)
  negf_torch.py      differentiable PyTorch implementation
verification/      correctness evidence (paper Sec. III-A, III-B)
  benchmarks_bw_tmm.py    Breit-Wigner benchmark + transfer-matrix cross-check
  gradient_families.py    autodiff vs. finite differences, 3 device families
  gradient_resonance.py   near-resonance analysis (flank / summit / Richardson)
  kwant_crosscheck.py     optional external cross-check (run on Colab; see below)
surrogate/         dataset + FNO/MLP study (paper Sec. III-C)
  generate_dataset.py     1000 in-family potential-transmission pairs
  generate_ood_splits.py  tall / narrow / 4-bump OOD splits
  train_fno_mlp.py        FNO + parameter-matched MLP, identical protocol
  evaluate_worstcase.py   worst-case metrics + paper figure
inverse_design/    verified inverse design (paper Sec. III-D)
  single_target_demo.py   flat-init resonance demo (double-barrier rediscovery)
  multiseed_sweep.py      3 targets x 5 seeds (checkpointed; re-run to resume)
  sweep_statistics.py     success statistics, failure-mode diagnosis
figures/           regenerate every paper figure
data/              dataset + OOD splits + precomputed sweep results (.npz)
```

## Install

```bash
pip install -r requirements.txt
```

Everything runs on CPU; a laptop or a free Google Colab instance suffices.

## Reproduce the paper

Each script prints the numbers quoted in the paper and can be run from any
directory:

```bash
python verification/benchmarks_bw_tmm.py    # 2.2e-16 (BW), 1.2e-10 (TMM)
python verification/gradient_families.py    # Table II rows
python verification/gradient_resonance.py   # flank 5.5e-8, summit analysis
python surrogate/generate_ood_splits.py     # OOD splits (or use data/)
python surrogate/train_fno_mlp.py           # Table III (FNO vs matched MLP)
python surrogate/evaluate_worstcase.py      # worst-case metrics + Fig. 6
python inverse_design/single_target_demo.py # Fig. 7 (double-barrier rediscovery)
python inverse_design/multiseed_sweep.py    # 15 runs, checkpointed (rerun to resume)
python inverse_design/sweep_statistics.py   # Table IV
python figures/make_figs.py                 # solver/validation figures
```

Precomputed results for the multi-seed sweep ship in `data/gap4_state.npz`,
so `sweep_statistics.py` works out of the box without re-running the sweep.

### Optional: Kwant cross-check

`verification/kwant_crosscheck.py` reproduces the transfer-matrix cross-check
with the independent [Kwant](https://kwant-project.org) package. Kwant has no
wheel for very recent Python versions; the script is written for Google Colab
(`pip install kwant` there) and is provided for readers who prefer an
established external reference implementation.

## Cite

If you use this code or dataset, please cite the paper (BibTeX to be added
with the archived DOI at camera-ready).

## License

MIT (see `LICENSE`).
