# Numerical Comparison of TQY, REMB, and REMD on the Positive Orthant

This folder contains the MATLAB implementation for the numerical experiment on the **positive orthant** equipped with the logarithmic metric.

The experiment compares the numerical performance of three algorithms for solving an equilibrium problem:

1. **TQY** — Adaptive extragradient-type method.
2. **REMB** — Regularized Extragradient Method with Busemann regularization.
3. **REMD** — Regularized Extragradient Method with distance regularization.

The numerical experiment is performed in dimension

\[
N=100.
\]

---

## 1. Mathematical Setting

The feasible space is the positive orthant

\[
\mathbb{R}_{++}^{N}
=
\{x=(x_1,\ldots,x_N)\in\mathbb{R}^{N}:x_i>0,\ i=1,\ldots,N\}.
\]

The space is equipped with the logarithmic metric

\[
d(x,y)
=
\left(
\sum_{i=1}^{N}
\left[
\log\left(\frac{x_i}{y_i}\right)
\right]^2
\right)^{1/2}.
\]

This metric is induced by the logarithmic coordinate transformation and is the distance used to measure the convergence of the algorithms.

---

## 2. Equilibrium Problem

The experiment considers the bifunction

\[
F(x,y)
=
\sum_{i=1}^{N}
\log(x_i)
\log\left(\frac{y_i}{x_i}\right).
\]

The corresponding equilibrium problem is to find \(p\in\mathbb{R}_{++}^{N}\) such that

\[
F(p,y)\geq 0,
\qquad
\forall y\in\mathbb{R}_{++}^{N}.
\]

For the numerical experiment, the reference solution is chosen as

\[
p=(1,\ldots,1)\in\mathbb{R}_{++}^{N}.
\]

Indeed, since \(\log(1)=0\),

\[
F(p,y)=0
\]

for every \(y\in\mathbb{R}_{++}^{N}\).

---

## 3. Initial Point

The initial point is generated randomly according to

```matlab
x0 = randi([5, 20], N, 1);
```

Thus, each component of the initial point is an integer between \(5\) and \(20\), ensuring that

\[
x_0\in\mathbb{R}_{++}^{N}.
\]

The dimension used in the experiment is

\[
N=100.
\]

The main problem parameters are

```matlab
N       = 100;
x0      = randi([5, 20], N, 1);
p       = ones(N, 1);
epsilon = 1e-8;
max_itr = 1000;
```

---

## 4. Algorithms Compared

### 4.1 REMB

The Regularized Extragradient Method with Busemann regularization is implemented using the resolvent

\[
J_t(x)=x^{\frac{1}{1+t}}
\]

componentwise.

The first step is

\[
y_k=J_{\lambda}(x_k),
\]

followed by

\[
x_{k+1}
=
J_F(y_k,x_k,\lambda).
\]

The corresponding mapping is

\[
J_F(x,y,t)
=
y\odot x^{-t},
\]

where the operations are understood componentwise.

The MATLAB implementation is:

```matlab
lambda_REMB = lambda;
x_REMB = x0;

for k = 1:max_itr

    y_REMB = J(x_REMB, lambda_REMB);

    x_REMB = JF(y_REMB, x_REMB, lambda_REMB);

    error_REMB(k) = d(p, x_REMB);

    if error_REMB(k) <= epsilon
        error_REMB = error_REMB(1:k);
        break;
    end

end
```

---

### 4.2 REMD

The Regularized Extragradient Method with distance regularization uses the same general two-step structure, but the first resolvent is evaluated with

\[
\frac{\lambda}{2}.
\]

Thus,

\[
y_k
=
J_{\lambda/2}(x_k),
\]

and

\[
x_{k+1}
=
J_F(y_k,x_k,\lambda).
\]

The implementation is:

```matlab
lambda_REMD = lambda;
x_REMD = x0;

for k = 1:max_itr

    y_REMD = J(x_REMD, lambda_REMD / 2);

    x_REMD = JF(y_REMD, x_REMD, lambda_REMD);

    error_REMD(k) = d(p, x_REMD);

    if error_REMD(k) <= epsilon
        error_REMD = error_REMD(1:k);
        break;
    end

end
```

---

### 4.3 TQY

The TQY algorithm uses an adaptive stepsize.

The initial parameters are

```matlab
lambda = 0.18;
delta  = 0.1;
```

At iteration \(k\), the adaptive parameters are

\[
\xi_k
=
1+\frac{1}{k^3},
\]

and

\[
\sigma_k
=
\frac{1}{(k+1000)^2}.
\]

The two extragradient steps are

\[
y_k
=
J_F(x_k,x_k,\lambda_k),
\]

and

\[
x_{k+1}
=
J_F(y_k,x_k,\lambda_k).
\]

The quantity

\[
D_k
=
F(x_k,x_{k+1})
-
F(x_k,y_k)
-
F(y_k,x_{k+1})
\]

is then used in the adaptive stepsize rule.

If \(D_k>0\), define

\[
L_1
=
\frac{
\delta\,
d(x_k,y_k)\,
d(x_{k+1},y_k)
}{
D_k
},
\]

and

\[
L_2
=
\xi_k\lambda_k+\sigma_k.
\]

Then,

\[
\lambda_{k+1}
=
\min\{L_1,L_2\}.
\]

If \(D_k\leq0\), the update becomes

\[
\lambda_{k+1}
=
\xi_k\lambda_k+\sigma_k.
\]

The corresponding MATLAB implementation is:

```matlab
xi_k    = 1 + 1 / k^3;
sigma_k = 1 / (k + 1000)^2;

y_TQY = JF(x_TQY, x_TQY, lambda_TQY);

x_TQY = JF(y_TQY, x_TQY, lambda_TQY);

D = F(x_prev, x_TQY) ...
    - F(x_prev, y_TQY) ...
    - F(y_TQY, x_TQY);

if D > 0

    L_1 = delta * d(x_prev, y_TQY) ...
        * d(x_TQY, y_TQY) / D;

    L_2 = xi_k * lambda_TQY + sigma_k;

    lambda_TQY = min(L_1, L_2);

else

    lambda_TQY = xi_k * lambda_TQY + sigma_k;

end
```

---

## 5. Parameters

The parameters used in the experiment are summarized below.

| Parameter | Value |
|---|---:|
| Dimension \(N\) | 100 |
| Initial point | Random integers in \([5,20]\) |
| Reference point \(p\) | \((1,\ldots,1)\) |
| Initial \(\lambda\) | 0.18 |
| \(\delta\) | 0.1 |
| Stopping tolerance | \(10^{-8}\) |
| Maximum iterations | 1000 |

---

## 6. Stopping Criterion

The convergence of the algorithms is monitored using the logarithmic distance from the reference solution \(p\):

\[
\operatorname{Er}(k)
=
d(p,x_k).
\]

The iteration terminates when

\[
d(p,x_k)
\leq 10^{-8}.
\]

In MATLAB:

```matlab
error_TQY(k) = d(p, x_TQY);

if error_TQY(k) <= epsilon
    error_TQY = error_TQY(1:k);
    break;
end
```

The same criterion is used for REMB and REMD.

---

## 7. Auxiliary Functions

### Logarithmic resolvent

The function

```matlab
J(x,t)
```

implements

\[
J_t(x)
=
x^{1/(1+t)}
\]

componentwise:

```matlab
function j = J(x, t)

    j = x .^ (1 / (1 + t));

end
```

---

### Proximal/resolvent mapping

The function

```matlab
JF(x,y,t)
```

implements

\[
J_F(x,y,t)
=
y\odot x^{-t}.
\]

```matlab
function jf = JF(x, y, t)

    jf = y .* x .^ (-t);

end
```

---

### Bifunction

The bifunction is

\[
F(x,y)
=
\sum_{i=1}^{N}
\log(x_i)
\log\left(\frac{y_i}{x_i}\right).
\]

It is implemented as

```matlab
function value = F(x, y)

    value = sum(log(x) .* log(y ./ x));

end
```

---

### Logarithmic distance

The metric

\[
d(x,y)
=
\sqrt{
\sum_{i=1}^{N}
\left[
\log\left(\frac{x_i}{y_i}\right)
\right]^2
}
\]

is implemented by

```matlab
function value = d(x, y)

    value = sqrt(sum((log(x ./ y)).^2));

end
```

All operations are componentwise, and positivity of the iterates is preserved by the algorithmic mappings.

---

## 8. Numerical Results

After running the script, MATLAB displays a table containing:

- the algorithm name,
- the number of iterations,
- the final error, and
- the computational time.

The output has the form

```text
==============================================================
                     NUMERICAL RESULTS
==============================================================
Method       Iterations      Final Error       Time (sec)
--------------------------------------------------------------
TQY          ...
REMB         ...
REMD         ...
==============================================================
```

The **Final Error** is the value

\[
d(p,x_k)
\]

at the final iteration.

The **Iterations** column records the number of iterations required to satisfy the stopping criterion.

The **Time (sec)** column gives the elapsed CPU time measured using MATLAB's `tic` and `toc` commands.

---

## 9. Convergence Plot

The script generates a semilogarithmic convergence plot of

\[
\operatorname{Er}(k)=d(p,x_k)
\]

against the iteration number \(k\).

The three curves correspond to:

- TQY,
- REMB,
- REMD.

The plot is generated using

```matlab
figure;

semilogy(1:length(error_TQY), ...
         error_TQY, ...
         'LineWidth', 2);
hold on;

semilogy(1:length(error_REMB), ...
         error_REMB, ...
         'LineWidth', 2);

semilogy(1:length(error_REMD), ...
         error_REMD, ...
         'LineWidth', 2);

hold off;
```

The resulting figure is saved as

```text
figure.eps
```

using

```matlab
print(gcf, 'figure.eps', '-depsc');
```

This EPS file can be directly incorporated into a LaTeX manuscript.

---

## 10. How to Run

Open MATLAB and navigate to the directory containing the script.

Run the main MATLAB file:

```matlab
main_comparison.m
```

The script will automatically:

1. Generate the initial point.
2. Set the problem and algorithm parameters.
3. Execute TQY.
4. Execute REMB.
5. Execute REMD.
6. Compute the convergence errors.
7. Display the numerical comparison.
8. Generate the convergence plot.
9. Save the plot as `figure.eps`.

No external data files are required.

---

## 11. Reproducibility

The initial point is generated randomly using

```matlab
x0 = randi([5, 20], N, 1);
```

Consequently, the exact numerical results may vary slightly between different runs.

For reproducibility, a fixed random seed may be specified before generating the initial point:

```matlab
rng(1);
x0 = randi([5, 20], N, 1);
```

This makes the initial point and the resulting numerical experiment deterministic.

---

## 12. Computational Environment

The code is written in MATLAB and uses standard MATLAB functions only.

The script is compatible with modern MATLAB versions supporting local functions in scripts.

---

## 13. Relation to the Paper

This numerical experiment corresponds to the experiment on the **positive orthant with the logarithmic metric** reported in the paper.

It is intended to provide a numerical comparison of the proposed regularized extragradient methods, REMB and REMD, with the TQY method.

The experiment is performed in dimension \(N=100\) and uses the logarithmic distance

\[
d(x,y)
=
\left(
\sum_{i=1}^{N}
\left[
\log\left(\frac{x_i}{y_i}\right)
\right]^2
\right)^{1/2}
\]

to measure the distance of the iterates from the reference solution.

---