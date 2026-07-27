# Sampling-overhead bound for the cardinality LCU

This note derives the bound used by `modules/lcu_constraint_flattener.py`. It documents the mathematical decomposition implemented there; it does **not** establish that sampling one branch reproduces the coherent penalty unitary on an individual shot.

## 1. Cardinality penalty on the Hamming-weight subspace

Let `N` be the number of decision qubits, let

\[
\hat n = \sum_{i=1}^{N} \frac{I - Z_i}{2},
\]

and define the target cardinality penalty

\[
f(k) = (k-k_\star)^2, \qquad k \in \{0,\ldots,N\}.
\]

The diagonal phase operation is

\[
U_f = \exp[-i\gamma f(\hat n)].
\]

On a computational-basis state of Hamming weight `k`, this operation has phase

\[
g(k) = \exp[-i\gamma(k-k_\star)^2].
\]

Because `g` is only required on the `W=N+1` possible Hamming weights, it has an exact discrete Fourier expansion on the cyclic group of size `W`.

## 2. Discrete Fourier expansion

Set

\[
\theta_j = \frac{2\pi j}{W}, \qquad
c_j = \frac{1}{W}\sum_{k=0}^{W-1}g(k)e^{-i\theta_j k},
\qquad j=0,\ldots,W-1.
\]

These are exactly the computations in `sample_lcu_branch`:

```python
coeff_sum += exp(-1j * gamma * penalty) * exp(-1j * theta_j * k_val)
c_coeffs[j] = coeff_sum / (num_qubits + 1)
```

By inverse discrete Fourier transform,

\[
g(k) = \sum_{j=0}^{W-1}c_j e^{i\theta_j k}.
\]

Therefore, on the Hamming-weight subspace,

\[
U_f = \sum_{j=0}^{W-1} c_j V_j,
\qquad
V_j = e^{i\theta_j\hat n}.
\]

Up to a global phase and the sign convention for `\theta_j`, each `V_j` is a product of identical single-qubit `R_z` rotations. This is the basis for `build_lcu_constraint_layer`, which applies the same sampled angle to every qubit. The code is a **sampled-basis approximation**: it chooses one `j`; it does not prepare the ancilla-assisted coherent linear combination above.

## 3. The \(\Gamma \le N+1\) bound

In LCU terminology, the coefficient 1-norm is

\[
\Gamma = \sum_{j=0}^{W-1}|c_j|.
\]

Since `g(k)` is a phase, `|g(k)|=1`. Applying the triangle inequality to each Fourier coefficient gives

\[
|c_j|
= \left|\frac{1}{W}\sum_{k=0}^{W-1}g(k)e^{-i\theta_j k}\right|
\le \frac{1}{W}\sum_{k=0}^{W-1}|g(k)|
= 1.
\]

There are `W=N+1` coefficients, so

\[
\boxed{\Gamma = \sum_{j=0}^{W-1}|c_j| \le W = N+1.}
\]

This is the conservative sampling-overhead bound asserted by the implementation. For the probability rule in the code,

\[
q_j = \frac{|c_j|}{\Gamma},
\]

normalization follows whenever `\Gamma > 0`, which holds because the inverse DFT of the nonzero phase vector `g` cannot have all coefficients equal to zero.

A tighter generic consequence of Parseval is also available with this normalization:

\[
\sum_j |c_j|^2 = 1
\quad\Longrightarrow\quad
\Gamma \le \sqrt{N+1}.
\]

The implementation and surrounding documentation intentionally report the looser but immediately derived `N+1` upper bound. Neither bound, by itself, determines the number of samples required for a chosen downstream estimator or establishes a hardware wall-clock advantage.

## 4. Relation to LCU literature

The decomposition above uses the same linear-combination idea as LCU Hamiltonian-simulation methods: represent a target operation as a weighted sum of efficiently implementable unitaries, with the coefficient norm governing normalization or sampling overhead. The implementation adapts only this diagonal, finite-Fourier specialization for a cardinality phase; it does **not** implement the oblivious-amplitude-amplification, ancilla preparation, or coherent select-oracle machinery of a full LCU simulation algorithm.

Primary reference:

- Andrew M. Childs and Nathan Wiebe, [*Hamiltonian simulation using linear combinations of unitary operations*](https://arxiv.org/abs/1202.5822), *Quantum Information & Computation* **12** (2012), 901–924.

For the modern LCU/block-encoding framing and its query-complexity interpretation, see:

- Dominic W. Berry, Andrew M. Childs, Richard Cleve, Robin Kothari, and Rolando D. Somma, [*Simulating Hamiltonian dynamics with a truncated Taylor series*](https://arxiv.org/abs/1412.4687), *Physical Review Letters* **114** (2015), 090502.

## 5. Scope and validation needed

The derivation proves an exact Fourier identity on the `N+1` Hamming-weight values and an upper bound on the associated coefficient 1-norm. It does not prove that the one-branch reservoir procedure preserves penalty expectation values, route quality, or noise resilience. Those are empirical claims that require comparison with coherent LCU, classical baselines, and hardware-noise experiments.
