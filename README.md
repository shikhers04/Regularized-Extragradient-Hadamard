# Regularized Extragradient Methods on Hadamard Manifolds

This repository contains the MATLAB codes used for the numerical
experiments associated with the paper

**"Regularized Extragradient Methods for Solving Equilibrium Problems
on Hadamard Manifolds."**

## Contents

The repository contains the following MATLAB files:

- `Main_Comparision.m`  
  Main script for comparing the numerical performance of the algorithms.

- `Neto_extragradient_step.m`  
  Implementation of the extragradient procedure used to solve the
  regularized equilibrium subproblems.

- `TQY_extragradient.m`  
  Implementation of the TQY extragradient-type algorithm used for comparison.

## Algorithms

The numerical experiments compare:

1. **REMB** — Regularized Extragradient Method with Busemann regularization.
2. **REMD** — Regularized Extragradient Method with distance regularization.
3. **TQY** — Extragradient-type method used as a comparison algorithm.

## Requirements

The codes are written in MATLAB.

Some numerical experiments require the MATLAB Optimization Toolbox.

## How to Run

Download or clone the repository and run

```matlab
Main_Comparision
