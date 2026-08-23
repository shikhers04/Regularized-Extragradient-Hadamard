# Example 4: Comparison of REMB, REMD, and TQY on Hyperbolic Space \(H^N\)

This folder contains the MATLAB implementation used for **Example 4** in the paper to compare the numerical performance of the following three algorithms for solving an equilibrium problem on the \(N\)-dimensional hyperbolic space:

1. **REMB** — Regularized Extragradient Method with Busemann regularization.
2. **REMD** — Regularized Extragradient Method with distance regularization.
3. **TQY** — Adaptive extragradient-type method.

The experiment is performed on the hyperboloid model of the hyperbolic space \(H^N\).

---

## 1. Hyperbolic Space

The \(N\)-dimensional hyperbolic space is represented using the hyperboloid model

\[
H^N
=
\left\{
x=(x_1,\ldots,x_N,x_{N+1})\in\mathbb{R}^{N+1}
:
\langle x,x\rangle_L=-1,\;
x_{N+1}>0
\right\},
\]

where the Lorentz inner product is defined by

\[
\langle x,y\rangle_L
=
\sum_{i=1}^{N}x_i y_i
-
x_{N+1}y_{N+1}.
\]

For \(x,y\in H^N\), the hyperbolic distance is given by

\[
d_H(x,y)
=
\operatorname{arcosh}
\left(
-\langle x,y\rangle_L
\right).
\]

In the numerical experiment, the dimension is set to

\[
N=50.
\]

Thus, the computational points belong to \(H^{50}\subset\mathbb{R}^{51}\).

---

## 2. Reference Point and Initial Point

A random initial point \(x_0\in H^N\) is generated as follows:

```matlab
w = randn(N,1);
x_last = sqrt(1 + sum(w.^2));
x0 = [w; x_last];
```

This construction guarantees

\[
\sum_{i=1}^{N}x_i^2-x_{N+1}^2=-1,
\]

with \(x_{N+1}>0\), and hence \(x_0\in H^N\).

The reference point used in the experiment is

\[
a=(0,\ldots,0,1)\in H^N.
\]

It is generated in MATLAB by

```matlab
a_ref = zeros(N+1,1);
a_ref(end) = 1;
```

---

## 3. Bifunction

The numerical experiment uses the bifunction

\[
F(x,y)
=
\frac12
\left[
d_H(y,a)^2-d_H(x,a)^2
\right],
\]

where \(a\in H^N\) is the reference point.

The corresponding MATLAB implementation is

```matlab
function value = F(x, y, a)

    dx = dH(x, a);
    dy = dH(y, a);

    value = 0.5 * (dy^2 - dx^2);

end
```

This bifunction is also used to calculate the adaptive quantity required in the TQY method.

---

## 4. Algorithms Compared

### 4.1 REMB

The Regularized Extragradient Method with Busemann regularization is implemented through two successive steps.

First,

\[
y_k
=
J_{\lambda_k}^{B}(x_k),
\]

where \(J_{\lambda_k}^{B}\) denotes the Busemann-type resolvent.

Then,

\[
x_{k+1}
=
J_{\lambda_k}^{F}(y_k).
\]

The regularization parameter is chosen as

\[
\lambda_k
=
1.1+\frac{1}{(k+1000)^2}.
\]

In the code:

```matlab
lambda_k = 1.1 + 1 / (k + 1000)^2;

y_REMB = J_hyperbolic(x_REMB, a_ref, lambda_k);

x_REMB = JF(y_REMB, a_ref, lambda_k);
```

---

### 4.2 REMD

The Regularized Extragradient Method with distance regularization is implemented similarly.

The first step uses the distance-type resolvent

\[
y_k=K_{\lambda_k}(x_k,a),
\]

followed by

\[
x_{k+1}
=
J_{\lambda_k}^{F}(y_k).
\]

The same regularization sequence

\[
\lambda_k
=
1.1+\frac{1}{(k+1000)^2}
\]

is used.

The corresponding MATLAB commands are

```matlab
lambda_k = 1.1 + 1 / (k + 1000)^2;

y_REMD = K(x_REMD, a_ref, lambda_k);

x_REMD = JF(y_REMD, a_ref, lambda_k);
```

---

### 4.3 TQY

The TQY method is implemented using the parameters

```matlab
lambda_TQY = 0.1;
delta = 0.1;
chi = 1.5;
```

The adaptive parameters are

\[
\xi_k=1+\frac{1}{k^2},
\qquad
\sigma_k=\frac{1}{(k+1000)^2}.
\]

The two iterative steps are

```matlab
y_TQY = JF(x_TQY, a_ref, lambda_TQY);

x_TQY = JF(y_TQY, a_ref, chi * lambda_TQY);
```

The adaptive parameter \(\lambda_k\) is subsequently updated according to the TQY scheme using

\[
D_k
=
F(x_k,x_{k+1})
-
F(x_k,y_k)
-
F(y_k,x_{k+1}).
\]

For the bifunction used in this experiment, the terms in \(D_k\) telescope theoretically. Nevertheless, \(D_k\) is explicitly evaluated in the implementation so that the code can readily be adapted to other bifunctions.

---

## 5. Stopping Criterion

For all three algorithms, the iteration is terminated when the distance between two consecutive iterates satisfies

\[
d_H(x_{k+1},x_k)
\leq 10^{-8}.
\]

The maximum number of iterations is

\[
\texttt{max\_itr}=300.
\]

The parameters are specified by

```matlab
tolerance = 1e-8;
max_itr = 300;
N = 50;
```

The quantity

\[
d_H(x_{k+1},x_k)
\]

is stored at every iteration and is used both for the stopping criterion and for the convergence plot.

---

## 6. Files and Functions

The main MATLAB script contains the complete numerical experiment, including the local functions required by the three algorithms.

The principal functions are:

| Function | Purpose |
|---|---|
| `J_hyperbolic` | Busemann-type hyperbolic resolvent used in REMB |
| `K` | Distance-type resolvent used in REMD |
| `JF` | Proximal/resolvent mapping associated with the bifunction |
| `F` | Bifunction used in the experiment |
| `dH` | Hyperbolic distance on \(H^N\) |
| `lorentz` | Lorentz inner product |
| `projectHN` | Normalization/projection onto the hyperboloid |

The script is self-contained and does not require external MATLAB toolboxes beyond standard MATLAB functionality.

---

## 7. Running the Experiment

Open MATLAB and navigate to the folder containing the script.

Run the main script:

```matlab
main_comparison.m
```

The script automatically:

1. Sets the numerical parameters.
2. Generates a random initial point in \(H^{50}\).
3. Defines the reference point.
4. Runs REMB.
5. Runs REMD.
6. Runs TQY.
7. Records the error at every iteration.
8. Displays the final iterates.
9. Displays the number of iterations, final error, and computational time.
10. Generates a convergence plot.

---

## 8. Numerical Output

The command window displays the final iterates obtained by the three algorithms:

```text
FINAL ITERATES

REMB:
...

REMD:
...

TQY:
...
```

It also reports the numerical performance:

```text
Algorithm       Iterations      Final Error          Time (sec)
----------------------------------------------------------------
REMB            ...
REMD            ...
TQY             ...
```

Here:

- **Iterations** = number of iterations required to satisfy the stopping criterion.
- **Final Error** = \(d_H(x_{k+1},x_k)\) at termination.
- **Time (sec)** = CPU time required by the corresponding algorithm.

Because the initial point is generated using `randn`, the numerical results may vary slightly between different runs.

---

## 9. Convergence Plot

The script generates a semilogarithmic plot of

\[
d_H(x_{k+1},x_k)
\]

against the iteration number.

The plot compares the convergence behavior of

- REMB,
- REMD, and
- TQY.

The relevant MATLAB command is

```matlab
semilogy(1:itr_REMB, errors_REMB, ...);
semilogy(1:itr_REMD, errors_REMD, ...);
semilogy(1:itr_TQY, errors_TQY, ...);
```

The vertical axis therefore represents the logarithmically scaled consecutive-iterate error

\[
d_H(x_{k+1},x_k).
\]

A decreasing curve indicates that the iterates are becoming progressively closer in the hyperbolic metric.

---

## 10. Important Implementation Details

### Hyperboloid normalization

Numerical round-off may cause a computed point to deviate slightly from the constraint

\[
\langle x,x\rangle_L=-1.
\]

Therefore, the function `projectHN` normalizes the point according to

\[
z
\leftarrow
\frac{z}{\sqrt{-\langle z,z\rangle_L}}.
\]

The upper sheet is selected by ensuring that the final coordinate is positive.

### Numerical safeguards

The implementation includes numerical safeguards in the computation of inverse hyperbolic functions and square roots. For example,

```matlab
argument = max(argument, 1);
```

is used before evaluating `acosh`, since theoretically

\[
-\langle x,y\rangle_L\geq 1
\]

for \(x,y\in H^N\).

These safeguards are included only to reduce the effects of floating-point arithmetic.

---

## 11. Reproducibility

Since the initial point is randomly generated using

```matlab
w = randn(N,1);
```

different runs may produce different initial points and, consequently, slightly different numerical results.

For exactly reproducible results, a random seed can be specified before generating the initial point, for example:

```matlab
rng(1);
```

This line can be placed before

```matlab
w = randn(N,1);
```

---

## 12. Parameters Used in Example 4

| Parameter | Value |
|---|---:|
| Dimension \(N\) | 50 |
| Maximum iterations | 300 |
| Stopping tolerance | \(10^{-8}\) |
| REMB/REMD \(\lambda_k\) | \(1.1+\frac{1}{(k+1000)^2}\) |
| TQY initial \(\lambda\) | 0.1 |
| TQY \(\delta\) | 0.1 |
| TQY \(\chi\) | 1.5 |
| Reference point | \((0,\ldots,0,1)\) |

---

## 13. Relation to the Paper

This code corresponds to **Example 4** in the paper and is provided to facilitate the reproduction and verification of the numerical comparison between REMB, REMD, and TQY in a negatively curved setting.

The experiment demonstrates the numerical behavior of the considered methods on the hyperbolic space \(H^{50}\), using the hyperbolic distance and Lorentzian geometry throughout the computations.

---

## 14. MATLAB Version

The code uses local functions within a MATLAB script and therefore requires:

**MATLAB R2016b or later.**

Newer MATLAB versions are recommended.

---