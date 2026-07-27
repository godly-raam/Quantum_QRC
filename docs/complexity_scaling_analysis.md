# Complexity and Scaling Analysis: Fourier-Based LCU vs. QAOA

## 1. The Bottleneck of Standard Quantum Constraints
In standard quantum optimization algorithms (like QAOA or QAOE), enforcing a global cardinality constraint (e.g., limiting the number of active delivery vehicles to $k$) requires encoding a quadratic penalty function: $f(x) = (\sum x_i - k)^2$.

When mapped to a quantum Hamiltonian, this requires all-to-all $ZZ$ interactions between every qubit representing a delivery node. 
* **CNOT Complexity:** $\mathcal{O}(N^2)$ entangling gates.
* **Circuit Depth (Heavy-Hex Topology):** Due to the limited connectivity of NISQ-era superconducting chips (like IBM's heavy-hex lattice), executing all-to-all interactions requires extensive routing via `SWAP` gates. This causes the circuit depth to scale as $\mathcal{O}(N^2)$.
* **Decoherence Risk:** High. The state is highly vulnerable to T1/T2 relaxation and two-qubit gate errors.

## 2. Ancilla-Free LCU Constraint Flattening (Proposed Architecture)
Our Quantum Reservoir Computing (QRC) Logistics Engine bypasses this entanglement bottleneck by decomposing the global penalty unitary into a continuous Fourier expansion. 

Because we only require the classical readout of the reservoir (sampling high-quality bitstrings) rather than preserving the full coherent phase distribution, we can drop the ancilla control qubits entirely. The constraint unitary $V(\theta)$ is flattened into a diagonal basis of permutation-invariant single-qubit rotations: $V(\theta) = \bigoplus_{i=1}^N e^{-i\theta Z_i / 2}$.

* **CNOT Complexity:** $0$ entangling gates.
* **Circuit Depth (Any Topology):** $\mathcal{O}(1)$. The constraint is applied as a single, fully parallelizable layer of $R_z$ gates, completely invariant to the hardware's coupling map.
* **Decoherence Risk:** Minimal. State fidelity is preserved for the non-linear feature extraction in the reservoir layer.

## 3. The Algorithmic Trade-off: Polynomial Sampling
The proposed construction trades the $\mathcal{O}(N^2)$ circuit-depth bottleneck of a dense penalty construction for an $\mathcal{O}(1)$ layer of single-qubit rotations. Its cost is shifted to classical runtime: approximating the penalty requires sampling discrete branches of the Fourier expansion.

For a cardinality constraint on $N$ variables, the stated LCU norm is $\Gamma \le N + 1$, yielding an $\mathcal{O}(N)$ classical sampling overhead.
* **Circuit-depth trade-off:** $\mathcal{O}(N^2) \rightarrow \mathcal{O}(1)$.
* **Sampling overhead:** $\mathcal{O}(N)$, with $\Gamma \le N + 1$.

## 4. Limitations and Future Work
The asymptotic depth reduction does not establish a real wall-clock crossover. In particular, the point at which $\mathcal{O}(N)$ classical sampling beats $\mathcal{O}(N^2)$ routed quantum depth under realistic two-qubit gate error rates, compilation overheads, and hardware connectivity has not yet been established empirically. Measuring that crossover across devices and problem sizes is future work.

The practical value of this trade-off depends on the target hardware, noise characteristics, classical sampling implementation, and real-time routing requirements.
