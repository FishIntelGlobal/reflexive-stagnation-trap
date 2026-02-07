# The Reflexive Stagnation Trap

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Substack-orange.svg)](https://fishintelglobal.substack.com/)
[![DOI](https://img.shields.io/badge/Status-Preprint-yellow.svg)]()

**How AI-Driven Epistemic Convergence May Undermine Long-Run Economic Growth**

Companion code for Paper I in the Applied Probabilistic Systems series.

> *The Republic of Letters thrived on disagreement. AI offers agreement at scale. The question is whether we can preserve the discord that progress may require.*

---

## The Mechanism

AI improves individual productivity but accelerates cognitive convergence. When everyone uses similar models, exploratory diversity collapses. The growth function:

```
g(q) = λ₀(1 − qᵝ)ᵅ · Δ + E(q)
```

produces an **inverted-U trajectory**: growth rises with AI quality until a critical threshold q\*, then declines as innovation losses from epistemic convergence exceed efficiency gains.

Under representative parameters (α=1, β=2, c=0.5):

```
q* = (c/4)^(2/3) = 0.250 exactly
```

Peak growth is **18.75% above the zero-AI baseline** before declining.

---

## Simulations

| # | Simulation | Purpose | Key Result |
|---|-----------|---------|------------|
| 1 | Core Growth Function | Verify analytical predictions | Table 1: all 21 values match closed-form to 3dp |
| 2 | Sensitivity Analysis | Parameter robustness | Table 2: interior maximum across all 25 (α,β) combinations |
| 3 | Agent-Based Model | Decentralised dynamics | 50/50 seeds show inverted-U; empirical q\* = 0.335 ± 0.052 |
| 4 | Engineered Discord | Counterfactual intervention | +23.4% cumulative growth, trap onset delayed 16 periods |

Simulations 1–2 reproduce every number in the paper exactly. Simulations 3–4 confirm qualitative predictions from decentralised agent dynamics.

---

## Quick Start

### Requirements

- Python 3.8+
- NumPy
- SciPy

### Install and Run

```bash
git clone https://github.com/FishIntelGlobal/reflexive-stagnation-trap.git
cd reflexive-stagnation-trap
pip install numpy scipy
python reflexive_stagnation_trap.py
```

### Expected Output

```
  SIMULATION 1: Core Growth Function
  q* closed-form:  0.250000000000000
  q* numerical:    0.250001638617435

       q  Innovation  Efficiency    Growth  Phase
    0.05       0.998       0.112     1.109  Rising
    0.25       0.938       0.250     1.188  Peak
    0.95       0.098       0.487     0.585  Declining

  SIMULATION 4: Counterfactual — Engineered Discord
  Cumulative Growth     192.3          237.2
  Improvement             ---         +23.4%
  Trap onset delayed by 16 periods.
```

---

## Model Assumptions

| # | Assumption | Formal Statement | Economic Basis |
|---|-----------|-----------------|----------------|
| A1 | Convex Homogenisation | h(q) = qᵝ, β > 1 | Convergence accelerates at high adoption |
| A2 | Diversity-Dependent Innovation | λ(h) = λ₀(1−h)ᵅ, α > 0 | Hong & Page (2004) diversity theorem |
| A3 | Quality-Growth Feedback | dq/dt = φ · g | Successful economies invest more in AI |
| A4 | Diminishing Marginal Efficiency | E′(q) > 0, E″(q) ≤ 0 | Standard production theory |
| A5 | Innovation Dominance | ∃ q̄: L(q̄) > E′(q̄) | Innovation losses can exceed efficiency gains |

For the representative specification, A5 requires c < 2βλ₀Δ.

---

## Three Falsifiable Predictions

1. **Convergence Effect**: Industries with faster AI adoption exhibit higher cross-firm similarity in forecasts and research topics.
2. **Non-Monotonicity**: The productivity–AI relationship is inverted-U, not linear.
3. **Exploration Compression**: Innovation novelty scores decline as AI model concentration increases.

---

## Repository Structure

```
reflexive-stagnation-trap/
├── reflexive_stagnation_trap.py   # All four simulations
├── README.md                      # This file
└── LICENSE                        # MIT License
```

---

## Series

| Paper | Title | Status |
|-------|-------|--------|
| **I** | **The Reflexive Stagnation Trap** | **Published (Feb 2026)** |
| II | The Homogeneity Threshold | Published (Jan 2026) |
| III | The First Principles of Uncertainty | February 2026 |

---

## Citation

```bibtex
@article{gething2026reflexive,
  title     = {The Reflexive Stagnation Trap: How {AI}-Driven Epistemic 
               Convergence May Undermine Long-Run Economic Growth},
  author    = {Gething, Jason},
  year      = {2026},
  month     = {February},
  note      = {Paper I in the Applied Probabilistic Systems series},
  publisher = {FishIntel Global},
  address   = {Eastbourne, UK},
  url       = {https://github.com/FishIntelGlobal/reflexive-stagnation-trap}
}
```

---

## License

This code is released under the [MIT License](LICENSE).

The accompanying paper is © 2026 Jason Gething / FishIntel Global Ltd. All rights reserved.

---

## Contact

**Jason Gething**  
FishIntel Global, Eastbourne, UK  
jason@fishintelglobal.com

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Jason%20Gething-blue?logo=linkedin)](https://www.linkedin.com/in/jason-gething/)
