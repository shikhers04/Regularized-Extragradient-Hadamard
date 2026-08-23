# Regularized Extragradient Methods on Hadamard Manifolds

This repository contains MATLAB implementations of numerical experiments for regularized extragradient methods applied to equilibrium problems on Hadamard manifolds. The experiments compare:

- **REMB**: regularized extragradient method with Busemann regularization;
- **REMD**: regularized extragradient method with distance regularization; and
- **TQY/Yao**: an adaptive extragradient method.

The code covers the positive orthant with its logarithmic metric, a bounded Nash-Cournot model, and the hyperboloid model of hyperbolic space.

## Repository structure

### `BuseAndDistance/`

Studies the effect of the regularization parameter in a three-dimensional positive-orthant example.

- `Lambda_Buse.m` runs the Busemann-regularized iteration.
- `Lambda_Dmetric.m` runs the distance-metric-regularized iteration.

Each script tests 10 values of `lambda` from `0.03` to `0.30`, performs 30 trials per value, and uses a maximum of 1000 iterations with tolerance `1e-8`. It reports mean and standard deviation for iteration counts and execution time as LaTeX table rows. The convergence and boxplot sections are available as commented code.

### `PositiveOrthantComp/`

Compares REMB, REMD, and TQY in dimension `N = 100` on the positive orthant. The distance is

```text
d(x,y) = sqrt(sum((log(x_i/y_i))^2)).
```

- `CompareNdim.m` runs one comparison from a random initial point, reports iterations, final errors, and execution times, and writes `figure.eps`.
- `BoxCompareNdim.m` runs 30 trials for each of 10 initial regularization values, using the same initial point for all three methods in each trial. It creates six EPS boxplots for iteration counts and CPU times.

Both scripts use `epsilon = 1e-8` and at most 1000 iterations. The equilibrium bifunction is implemented as

```text
F(x,y) = sum(log(x_i) * log(y_i/x_i)).
```

The reference solution is the all-ones vector.

### `Nash_Courant/`

Implements a four-company Nash-Cournot equilibrium experiment on the bounded box

```text
[1000,2000] x [500,2500] x [800,1500] x [500,3000].
```

- `Main_Comparision.m` compares Busemann + Neto, distance + Neto, and TQY/Yao, then plots their convergence histories.
- `Neto_extragradient_step.m` provides the constrained two-step Neto extragradient procedure.
- `TQY_extragradient.m` provides the constrained adaptive TQY procedure.

The Nash-Cournot bifunction is

```text
F(x,y) = (P*x + Q*y + r)' * (y - x).
```

The outer methods use at most 1000 iterations; the inner Neto method uses at most 10 iterations. The constrained minimizations use `fmincon` with the interior-point algorithm.

### `Hyperbolic/`

Compares REMB, REMD, and TQY on the hyperboloid model of `H^50`, represented in `R^51` by

```text
H^N = {x : <x,x>_L = -1 and x_(N+1) > 0},
```

with Lorentz product

```text
<x,y>_L = sum(x_i*y_i) - x_(N+1)*y_(N+1)
```

and distance `d_H(x,y) = acosh(-<x,y>_L)`. `HyperComp.m` generates a random valid initial point, uses `(0,...,0,1)` as the reference point, runs at most 300 iterations with tolerance `1e-8`, prints a numerical comparison, and plots the consecutive-iterate errors.

The script includes local implementations of the Busemann resolvent, distance-type resolvent, proximal mapping, bifunction, hyperbolic distance, Lorentz product, and projection back onto the hyperboloid. Numerical safeguards protect the hyperboloid constraint from floating-point drift.

## Requirements

- MATLAB R2020b or later is recommended. The scripts use local functions in script files.
- The Nash-Cournot experiment requires the MATLAB Optimization Toolbox because it calls `fmincon`.
- The other experiments use MATLAB built-in numerical functions and do not require an additional toolbox.

## Running the experiments

Open MATLAB, set the current folder to the repository root, and run one of the scripts from its subfolder. For example:

```matlab
cd BuseAndDistance
Lambda_Buse
```

Other entry points are:

```matlab
cd PositiveOrthantComp
CompareNdim
BoxCompareNdim

cd ../Nash_Courant
Main_Comparision

cd ../Hyperbolic
HyperComp
```

The scripts display numerical results in the MATLAB command window and create figures or EPS files where indicated above. Results depend on the random initial points; `Lambda_Buse.m` and `Lambda_Dmetric.m` explicitly set the random seed to `1`, while the comparison scripts use MATLAB's current random-number state.

## Citation

These implementations accompany the paper:

**Regularized Extragradient Methods for Solving Equilibrium Problems on Hadamard Manifolds**

