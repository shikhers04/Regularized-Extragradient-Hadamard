# Nash–Cournot Numerical Experiment

This folder contains the MATLAB codes for the numerical Nash–Cournot equilibrium experiment in the paper:

**“Regularized Extragradient Methods for Solving Equilibrium Problems on Hadamard Manifolds.”**

The experiment compares three methods:

- **REMB** — Busemann + Neto
- **REMD** — Distance + Neto
- **TQY** — TQY/Yao extragradient method

The problem considers **four companies** on the positive orthant \(\mathbb{R}_{++}^4\).

## Files

- `Main_Comparision.m` — Main script for running the numerical experiment.
- `Neto_extragradient_step.m` — Neto extragradient procedure used by REMB and REMD.
- `TQY_extragradient.m` — TQY/Yao method used for comparison.

## Problem Setting

The feasible set is

\[
C=[1000,2000]\times[500,2500]\times[800,1500]\times[500,3000].
\]

The initial point is

\[
x_0=(570,948,503,812)^T.
\]

The logarithmic distance is

\[
d(x,y)=
\left(
\sum_{i=1}^{4}
\left[\log\left(\frac{x_i}{y_i}\right)\right]^2
\right)^{1/2}.
\]

The Nash–Cournot bifunction used in the experiment is

\[
F(x,y)=(Px+Qy+r)^T(y-x).
\]

## Parameters

- Dimension: \(N=4\)
- Maximum iterations: `1000`
- Maximum inner iterations: `10`
- REMB/REMD tolerance: \(10^{-8}\)
- Neto tolerance: \(10^{-2}\)
- TQY tolerance: \(10^{-8}\)

The regularization parameter is

\[
\lambda_k=0.8+\frac{1}{(k+1000)^2}.
\]

## How to Run

Make sure all three MATLAB files are in the same folder:

```text
Main_Comparision.m
Neto_extragradient_step.m
TQY_extragradient.m
```

Then run:

```matlab
Main_Comparision
```

The program displays:

- final iterates,
- number of iterations,
- final error,
- computational time,

and produces a convergence plot comparing REMB, REMD, and TQY.

## Requirements

- MATLAB
- MATLAB **Optimization Toolbox** (`fmincon` is used).

## Citation

If you use these codes, please cite the associated paper:

**“Regularized Extragradient Methods for Solving Equilibrium Problems on Hadamard Manifolds.”**

**Author:** Shikher Sharma