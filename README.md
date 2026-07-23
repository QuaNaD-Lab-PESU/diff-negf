# diff-negf

**Inverse Design of Quantum Transport with a Verified Differentiable NEGF Solver.**

Quantum and Nano Devices (QuaNaD) Lab, Department of Electronics and Communication
Engineering, PES University - Electronic City Campus, Bengaluru.

This repository accompanies the paper *"Inverse Design of Quantum Transport with a
Verified Differentiable NEGF Solver"* and contains the complete stack: solver,
differentiable implementation, verification scripts, the dataset with designated
out-of-family (OOD) splits, and scripts that regenerate every figure and table in
the paper.

## Layout

```
src/               core solvers
  negf_numpy.py      classical 1D coherent-NEGF (reference)
  negf_torch.py      differentiable PyTorch implementation
verification/      correctness evidence (paper Sec. IV-A, IV-B)
  benchmarks_bw_tmm.py    Breit-Wigner benchmark + transfer-matrix cross-check
  gradient_families.py    autodiff vs. finite differences, 3 device families
  gradient_resonance.py   near-resonance analysis (flank / summit / Richardson)
  kwant_crosscheck.py     optional external cross-check (run on Colab; see below)
surrogate/         dataset + FNO/MLP study (paper Sec. IV-C)
  generate_dataset.py     1000 in-family potential-transmission pairs
  generate_ood_splits.py  tall / narrow / 4-bump OOD splits
  train_fno_mlp.py        FNO + parameter-matched MLP, identical protocol
  evaluate_worstcase.py   worst-case metrics + paper figure
inverse_design/    verified inverse design (paper Sec. IV-D)
  single_target_demo.py   flat-init resonance demo (double-barrier rediscovery)
  multiseed_sweep.py      3 targets x 5 seeds (checkpointed; re-run to resume)
  sweep_statistics.py     success statistics, failure-mode diagnosis
figures/           regenerate every paper figure
data/              dataset + OOD splits + precomputed sweep results (.npz)
```

## Install

```
pip install -r requirements.txt
```

Everything runs on CPU; a laptop or a free Google Colab instance suffices.

## Reproduce the paper

Each script prints the numbers quoted in the paper and can be run from any
directory, and all figures are written to `figs/` at the repository root:

```
python figures/make_figs.py                 # Figs. 1-2, 4 (solver + gradient)
python figures/make_fig_verify.py           # Fig. 3 (Breit-Wigner + transfer matrix)
python figures/make_fig_multiseed.py        # Fig. 6 (multi-seed study)
python verification/benchmarks_bw_tmm.py    # Table III: 3.3e-16 (BW), 1.2e-10 (TMM)
python verification/kwant_crosscheck.py     # Table III: 1.17e-12 (Kwant; see below)
python verification/gradient_families.py    # Table IV rows + Fig. 4
python verification/gradient_resonance.py   # Table IV: flank 5.5e-8, summit analysis
python surrogate/generate_ood_splits.py     # OOD splits (or use data/)
python surrogate/train_fno_mlp.py           # Table V (FNO vs matched MLP)
python surrogate/evaluate_worstcase.py      # Table V worst-case metrics + Fig. 5
python inverse_design/single_target_demo.py # Fig. 7 (double-barrier rediscovery)
python inverse_design/multiseed_sweep.py    # 15 runs, checkpointed (rerun to resume)
python inverse_design/sweep_statistics.py   # Table VI
```

Precomputed results for the multi-seed sweep ship in
`data/multiseed_sweep_state.npz`, so `sweep_statistics.py` works out of the box
without re-running the sweep.

### Optional: Kwant cross-check

`verification/kwant_crosscheck.py` reproduces the double-barrier cross-check with
the independent [Kwant](https://kwant-project.org) package, for readers who
prefer an established external reference implementation. Measured agreement:
`max |T_Kwant - T_NEGF| = 1.17e-12` over 900 energies (Table III).

Kwant 1.5.0 ships pregenerated Cython sources that do not compile against
NumPy >= 2.0, so build it against NumPy 1.x:

```
python -m venv kwenv
kwenv/bin/pip install "numpy<2" scipy cython tinyarray
kwenv/bin/pip install --no-build-isolation kwant
kwenv/bin/python verification/kwant_crosscheck.py
```

Google Colab also works (`!pip install kwant`).

## Cite

If you use this code or dataset, please cite the paper (BibTeX to be added with
the archived DOI at camera-ready).

## License

MIT (see `LICENSE`).
