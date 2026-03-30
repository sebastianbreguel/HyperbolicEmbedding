# Experimental Protocol: Euclidean vs. Hyperbolic Neural Networks

**Project:** Hyperbolic Embedding — torchExperiments
**Research question:** When does hyperbolic geometry provide a measurable advantage
over Euclidean geometry for classification and regression on structured data?
**Last updated:** 2026-03-30

---

## 1. Overview

This document specifies the full experimental protocol for comparing Euclidean and
hyperbolic (Poincaré ball) neural networks across three task families:

| Task family | Dataset | Structure | Expected geometry winner |
|---|---|---|---|
| Ganea prefix classification | Synthetic prefix sequences | Explicit tree/hierarchy | Hyperbolic |
| Mircea phylogenetic regression | Synthetic binary tree | Ground-truth tree metric | Hyperbolic |
| MNIST classification | UMAP-reduced digits | No hierarchy (flat) | Neither (null control) |

The experiments are numbered 1–9 and designed to be executed sequentially.
Each later experiment builds on or is calibrated against the results of earlier ones.
Do not run Experiments 3–9 before Experiment 1 has completed and results have been
inspected.

---

## 2. Infrastructure

### 2.1 Execution

All experiments use the YAML-config system.  Run any experiment as:

```bash
uv run python main.py --config experiments/exp1_geometry_baseline.yaml
```

Preview a sweep without running:

```bash
uv run python main.py --config experiments/exp1_geometry_baseline.yaml --dry-run
```

Run a specific combination (0-indexed) from a sweep:

```bash
uv run python main.py --config experiments/exp1_geometry_baseline.yaml --experiment 0
```

Filter to a subset:

```bash
uv run python main.py --config experiments/exp1_geometry_baseline.yaml \
  --filter model=hyperbolic,optimizer=Radam
```

### 2.2 Data generation

Before running any Ganea experiment with a new `replace` value, generate the
corresponding CSV:

```bash
uv run python data.py --task ganea --replace <value>
```

For MNIST with a non-default `dimensions` value, regenerate the UMAP embeddings:

```bash
# Edit DIMENSIONS in config.py to the target value, then:
uv run python data.py --task MNIST
```

The Mircea phylogenetic data is regenerated automatically and does not require a
separate step.

### 2.3 Tracking

All experiments log to W&B project `hyperbolic-geometry-comparison` (set in each
YAML).  Use `wandb_project: hyperbolic-retrieval-pilot` for Experiment 9 to keep
retrieval results separate.

---

## 3. Statistical Analysis Plan

### 3.1 Primary metric per task

| Task | Primary metric | Secondary metric |
|---|---|---|
| Ganea | Test accuracy (fraction) | Training convergence speed (epochs to 90% val acc) |
| Mircea | Normalised L2 error (MSE / target variance) | Per-edge regression error |
| MNIST | Test accuracy (fraction) | Top-5 accuracy |

### 3.2 Significance testing

For each pairwise comparison (e.g., hyperbolic+Radam vs. Euclidean+Radam):

1. Collect the `runs` independent test-metric values (each run uses a different
   random initialisation but the same data split — verify that `seed` is varied
   across runs in the training loop, or add `seed: [0,1,...,9]` to the sweep if
   the trainer does not auto-increment seeds).
2. Run a two-sided Welch's t-test (does not assume equal variance).
3. Report: mean ± standard deviation, t-statistic, p-value, and Cohen's d effect
   size.
4. Use p < 0.05 as the threshold for claiming a statistically significant
   difference.  Apply Bonferroni correction when making multiple simultaneous
   comparisons within a single experiment (e.g., Exp 4 tests 3 dataset depths ×
   3 hidden sizes = 9 comparisons; threshold becomes p < 0.05 / 9 ≈ 0.0056).

### 3.3 Reporting failures

For Experiment 7 (SGD ablation), record runs where the training loss produces NaN
or the validation loss diverges (exceeds 2× the initial loss at any epoch).
Report the failure count alongside mean accuracy, not as exclusions.  A high
failure rate for hyperbolic+SGD is itself a scientifically meaningful result.

### 3.4 Effect size as the primary claim

p-values alone are insufficient.  The central claim is that hyperbolic geometry
provides a *practically significant* advantage on hierarchical data.  Report
Cohen's d and interpret:

- d < 0.2: negligible (likely noise)
- d 0.2–0.5: small but real
- d 0.5–0.8: medium (sufficient for a paper claim)
- d > 0.8: large (strong claim)

---

## 4. Experiments

### Execution order

```
Exp 1  (geometry baseline)          ← must complete first
  ├── Exp 2  (capacity efficiency)  ← can run in parallel with Exp 1
  ├── Exp 7  (optimizer ablation)   ← can run after Exp 1
Exp 3  (noise robustness)           ← run after Exp 1 validates geometry effect
Exp 4  (hierarchy depth)            ← run after Exp 1
Exp 5  (phylogenetic primary)       ← independent; can run any time
Exp 5b (phylogenetic capacity)      ← run after Exp 5 completes
Exp 6  (MNIST flat control)         ← independent; run any time
Exp 8  (MNIST dimensions)           ← run after Exp 6 to inform data prep
Exp 9  (retrieval pilot)            ← blocked until encode() is added to HNN
```

### Compute budget summary

| Experiment | File | Runs | Estimated wall time* |
|---|---|---|---|
| 1. Geometry baseline | exp1_geometry_baseline.yaml | 40 | ~2h |
| 2. Capacity efficiency | exp2_capacity_efficiency.yaml | 64 | ~3h |
| 3. Noise robustness | exp3_noise_robustness.yaml | 80 | ~4h |
| 4. Hierarchy depth | exp4_hierarchy_depth.yaml | 144 | ~7h |
| 5. Phylogenetic primary | exp5_phylogenetic.yaml | 40 | ~3h |
| 5b. Phylogenetic capacity | exp5b_phylogenetic_capacity.yaml | 48 | ~4h |
| 6. MNIST flat control | exp6_mnist_flat_baseline.yaml | 120 | ~4h |
| 7. Optimizer ablation | exp7_optimizer_ablation.yaml | 60 | ~3h |
| 8. MNIST dimensions | exp8_dimensions_mnist.yaml | 96 | ~4h |
| 9. Retrieval pilot (BLOCKED) | exp9_retrieval_pilot.yaml | 48 | ~3h |
| **Total** | | **740** | **~37h** |

*Wall time estimated at ~3 minutes per run on a single GPU. Adjust for your
hardware. Experiments 1–8 total ~692 unblocked runs (~34h).

---

### Experiment 1 — Geometry Baseline (Primary Comparison)

**File:** `experiments/exp1_geometry_baseline.yaml`
**Conditions:** 2 models × 2 optimizers = 4, each × 10 runs = 40 runs total
**Key hypotheses:** H1a (hyperbolic advantage on hierarchical data), H1b
(Riemannian optimizer necessary for hyperbolic), H1c (optimizer parity for
Euclidean).

**Decision gate:**
- If hyperbolic+Radam does NOT outperform Euclidean+Radam by at least 2 pp at
  p < 0.05, revisit the data generation (check that replace=0.1 data file exists
  and has no corruption) before proceeding to other experiments.
- If the Euclidean+Adam vs. Euclidean+Radam gap exceeds 3 pp, investigate whether
  ManifoldParameter separation in `obtain_optimizer` is working correctly for flat
  parameters.

**Analysis script targets:**
- Bar chart: mean test_acc ± std for all 4 conditions.
- Table: t-statistic and p-value for hyperbolic+Radam vs. Euclidean+Radam.
- This chart becomes Figure 1 in the paper.

---

### Experiment 2 — Capacity Efficiency

**File:** `experiments/exp2_capacity_efficiency.yaml`
**Conditions:** 2 models × 4 hidden_sizes = 8, each × 8 runs = 64 runs total
**Key hypothesis:** H2 (hyperbolic geometry is more parameter-efficient for
hierarchical data; a compact hyperbolic model matches a much larger Euclidean one).

**Analysis script targets:**
- Line chart: mean test_acc vs. hidden_size, one line per geometry.
- Identify the crossover point (smallest hidden_size where Euclidean catches up
  to hyperbolic).  Label this "minimum Euclidean capacity to match hyperbolic."
- Report parameter counts alongside hidden_size for interpretability:
  HNN has (in × hidden + hidden × out) parameters; compute these for each size.

---

### Experiment 3 — Noise Robustness

**File:** `experiments/exp3_noise_robustness.yaml`
**Conditions:** 2 models × 5 noise levels = 10, each × 8 runs = 80 runs total
**Key hypotheses:** H3a (advantage erodes with noise), H3b (hyperbolic degrades
more gracefully at moderate noise).

**Pre-run checklist:**
Generate all five noise-level data files before submitting the sweep:

```bash
for r in 0.0 0.2 0.4 0.6 0.8; do
    uv run python data.py --task ganea --replace $r
done
```

**Analysis script targets:**
- Line chart: accuracy vs. replace, one line per geometry.
- Mark the "crossover noise level" — the replace value at which the accuracy
  gap drops below 1 pp.  This is a key quantitative finding for the paper.
- Error bars must be shown (the variance increases near chance level).

---

### Experiment 4 — Hierarchy Depth

**File:** `experiments/exp4_hierarchy_depth.yaml`
**Conditions:** 2 models × 3 datasets × 3 hidden_sizes = 18, each × 8 runs = 144 runs
**Key hypothesis:** H4 (deeper hierarchies amplify the hyperbolic advantage).

**Confound note:**
The input dimensionality is not constant across dataset values (it is
`20 * LARGE + int(dataset * 0.2)` = 42, 46, 50 for datasets 10, 30, 50).
This is a minor confound that should be reported in the paper's methodology
section but is unavoidable given the feature encoding.

**Analysis script targets:**
- Heatmap: accuracy gap (hyperbolic − Euclidean) as a function of
  dataset × hidden_size.
- Line chart: gap vs. dataset for each hidden_size.

---

### Experiment 5 — Phylogenetic Regression (Primary)

**File:** `experiments/exp5_phylogenetic.yaml`
**Conditions:** 2 models × 2 optimizers = 4, each × 10 runs = 40 runs
**Key hypotheses:** H5a (hyperbolic advantage on real tree metric), H5b
(optimizer criticality on continuous regression).

**Analysis script targets:**
- Bar chart of normalised L2 error for all 4 conditions.
- Compare structure with Figure from Exp 1 — does the geometry advantage hold
  across task type (classification → regression)?

---

### Experiment 5b — Phylogenetic Capacity (Companion)

**File:** `experiments/exp5b_phylogenetic_capacity.yaml`
**Conditions:** 2 models × 3 hidden_sizes = 6, each × 8 runs = 48 runs
**Key hypothesis:** H2 replication on regression (capacity efficiency).

**Analysis script targets:**
- Line chart: normalised L2 error vs. hidden_size per geometry.
- Compare against Exp 2 capacity curve: is the efficiency ratio similar on
  regression and classification?

---

### Experiment 6 — MNIST Flat Control

**File:** `experiments/exp6_mnist_flat_baseline.yaml`
**Conditions:** 2 models × 3 dimensions × 2 hidden_sizes = 12, each × 10 runs = 120 runs
**Key hypothesis:** H6a (no geometry advantage on flat data — the null control).

**Pre-run checklist:**
Regenerate UMAP embeddings for each `dimensions` value:

```bash
# Set DIMENSIONS=10 in config.py, then:
uv run python data.py --task MNIST
mv data/MNIST/train.csv data/MNIST/train_d10.csv
mv data/MNIST/test.csv  data/MNIST/test_d10.csv

# Set DIMENSIONS=15 in config.py (this is the default — may already exist), then:
uv run python data.py --task MNIST
mv data/MNIST/train.csv data/MNIST/train_d15.csv
mv data/MNIST/test.csv  data/MNIST/test_d15.csv

# Set DIMENSIONS=30 in config.py, then:
uv run python data.py --task MNIST
mv data/MNIST/train.csv data/MNIST/train_d30.csv
mv data/MNIST/test.csv  data/MNIST/test_d30.csv
```

Alternatively, implement a `dimensions`-aware data loader path in `getMNIST()`
that reads `data/MNIST/train_d{d}.csv` when `config.DIMENSIONS == d`.

**Analysis script targets:**
- Side-by-side bar chart: accuracy for both geometries at each
  dimensions × hidden_size.
- Run paired t-tests for each pair; report that none are significant at p < 0.05.
- This "null result" section appears in the paper as "On flat data, geometry does
  not matter" — it is critical for falsifiability.

---

### Experiment 7 — Optimizer Ablation

**File:** `experiments/exp7_optimizer_ablation.yaml`
**Conditions:** 2 models × 3 optimizers = 6, each × 10 runs = 60 runs
**Key hypotheses:** H7a (SGD as lower bound), H7b (Riemannian necessity gap
larger for hyperbolic), H7c (SGD instability on hyperbolic manifold).

**Analysis note:**
Inspect W&B training curves, not just final accuracy.  Look for:
- Loss NaN or spike events (manifold boundary violations).
- Slow convergence in SGD runs.
- High run-to-run variance in SGD compared to Adam/Radam.

Report the *failure rate* (runs with NaN loss or validation loss diverging >2×
initial value) as a standalone metric in addition to accuracy.

---

### Experiment 8 — MNIST Dimensions Scaling

**File:** `experiments/exp8_dimensions_mnist.yaml`
**Conditions:** 2 models × 6 dimension values = 12, each × 8 runs = 96 runs
**Key hypothesis:** H8 (Euclidean improves monotonically with dimensions;
hyperbolic peaks early and may degrade at high dimensions due to ill-conditioning).

**Pre-run checklist:**
Requires UMAP data for all 6 dimension values: {5, 10, 15, 20, 30, 50}.
Generate each as described in Experiment 6's checklist.

**NOTE:** This experiment likely requires a code change to `getMNIST()` to support
dimension-specific CSV paths.  Implement before running.

---

### Experiment 9 — Retrieval Pilot (BLOCKED)

**File:** `experiments/exp9_retrieval_pilot.yaml`
**Status:** BLOCKED — requires `encode()` method on `HNN`.
**Conditions:** 2 models × 3 hidden_sizes = 6, each × 8 runs = 48 runs
**Key hypotheses:** H9a–H9c (hyperbolic embeddings improve nearest-neighbour
retrieval quality on hierarchical data).

**Prerequisites:**
1. Add `encode()` method to `models/hnn.py` (see YAML comment for the code).
2. Add retrieval evaluation to `training/metrics.py` or a new
   `training/retrieval_metrics.py` module implementing:
   - `recall_at_k(embeddings, labels, k, distance_fn)`
   - `mean_reciprocal_rank(embeddings, labels, distance_fn)`
   - `ndcg_at_k(embeddings, labels, k, relevance_fn, distance_fn)`
3. Implement Poincaré distance (`arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))`
   in `manifolds/poincare.py` as `dist(u, v)` if not already present.
4. Add a `--retrieval` flag to `main.py` (or a new `retrieval_eval.py` script)
   that extracts embeddings after training and computes retrieval metrics.

**See Section 5 for the full future retrieval research plan.**

---

## 5. Future Directions: Hyperbolic Retrieval

This section outlines a research agenda for hyperbolic nearest-neighbour
retrieval, motivated by the user's interest in retrieval tasks.  These
experiments require infrastructure beyond the current codebase but are
designed to be buildable on top of it.

### 5.1 Motivation

The current experiments evaluate geometry via end-to-end classification and
regression accuracy.  Retrieval tasks test a complementary property: whether
the *geometry of the embedding space* faithfully reflects the *structure of
the data* such that nearby points in embedding space are truly related items.

Hyperbolic geometry has two properties that make it attractive for retrieval:

1. **Exponential growth of volume:** A Poincaré ball of radius r contains
   exponentially more volume near the boundary than at the center.  This
   matches the exponential growth of node count in trees, so all leaf nodes
   can be embedded with small pairwise distances to their siblings.

2. **Low distortion of tree metrics:** Sarkar (2011) showed that any weighted
   tree can be embedded into the hyperbolic plane with arbitrarily low distortion
   using only 2 dimensions, whereas Euclidean space requires O(n) dimensions.
   Low-distortion embeddings directly translate to high-quality retrieval.

### 5.2 Near-term retrieval experiment (Exp 9 expansion)

**Once Exp 9 pilot validates H9a**, expand to a full retrieval study:

| Variable | Values | Rationale |
|---|---|---|
| Geometry | euclidean, hyperbolic | Primary comparison |
| Training objective | cross-entropy (current), triplet loss (new) | Does metric learning improve retrieval? |
| Distance function | L2 (Euclidean), Poincaré (hyperbolic) | Does using manifold-correct distance matter at inference? |
| Embedding dim | 2, 4, 8, 16 | Low-dim regime where hyperbolic advantage is largest |
| Dataset | ganea-30, mircea | Tree-structured data |

**Key metric:** Recall@1 and MRR on a held-out query set.  Ground truth = same
subtree membership.

### 5.3 Triplet loss for hyperbolic retrieval

To train embeddings specifically for retrieval, replace the classification head
with a hyperbolic triplet loss (Ermolov et al. 2022, "Hyperbolic Vision
Transformers"):

```
L_triplet = max(0, d_h(anchor, positive) - d_h(anchor, negative) + margin)
```

where `d_h` is Poincaré distance.  This requires:
1. A triplet sampler that uses tree structure to define positive/negative pairs.
2. A modified `HNN.encode()` that returns the penultimate embedding.
3. No classification head.

For the Ganea task, positives are (word, its prefix) pairs and negatives are
random non-prefix pairs (the existing data generation already encodes this).

### 5.4 Comparison against Poincaré embeddings baseline

Nickel & Kiela (2017) train embeddings directly on tree edges with a
reconstruction objective.  This is a strong retrieval-specific baseline
that the HNN+cross-entropy approach cannot be expected to beat at retrieval
without a retrieval-specific training signal.

Recommended baseline comparison table for a retrieval paper section:

| Model | Training objective | Retrieval? | Hierarchy aware? |
|---|---|---|---|
| Euclidean MLP | Cross-entropy | No (classification) | No |
| Hyperbolic HNN | Cross-entropy | Pilot (Exp 9) | Via geometry |
| Poincaré embeddings (NK 2017) | Tree edge reconstruction | Yes | Yes (explicit) |
| Hyperbolic HNN + triplet | Triplet loss | Yes | Via geometry + loss |

### 5.5 Scaling to real-world hierarchical retrieval

The long-term goal is to apply hyperbolic retrieval to real-world datasets with
known hierarchical structure:

- **WordNet subtree retrieval** (Nickel & Kiela 2017 benchmark): Given a word,
  retrieve its hypernyms/hyponyms.  Ground truth = WordNet edge graph.
- **Scientific paper retrieval** by topic hierarchy (CiteSeer, ACM taxonomy).
- **Biological taxonomy** (species retrieval in phylogenetic trees).

These require scaling the architecture (deeper HNNs, larger `hidden_size`) and
evaluating with standard IR metrics (MAP, NDCG@10, Recall@100).

---

## 6. Known Limitations and Confounds

### 6.1 Input dimensionality confound (Exp 4)

For the Ganea task, the model input size is `20 * LARGE + int(dataset * 0.2)`.
This means datasets 10, 30, and 50 have input sizes 42, 46, and 50 respectively.
The increasing input size is a minor confound: a slightly larger input already
provides more information regardless of geometry.  This should be disclosed in
the paper.  A fully controlled version would pad all inputs to the same size, but
this would require modifying `data_loader.py`.

### 6.2 Fixed curvature (c=1)

The current implementation uses a fixed curvature `c=1` for the Poincaré ball.
In principle, the optimal curvature is data-dependent (Gu et al. 2019, "Learning
Mixed-Curvature Representations").  Ablating over `c` values is a natural
extension but is not supported in the current codebase.  Disclose this as a
limitation.

### 6.3 Seed management

The current `config.py` has a single `SEED` constant.  Multiple independent runs
will produce identical results unless the training loop increments the seed per
run.  Verify that `main.py` either:
(a) auto-increments the seed for each run (`seed = SEED + run_index`), or
(b) does not set any seed (relying on different PyTorch random states per run).

If (b), the runs are stochastic but not reproducible.  For reproducibility,
option (a) is strongly preferred.  If neither is currently implemented, add
seed incrementing before proceeding with Experiments 1–9 to ensure your runs
are genuinely independent.

### 6.4 UMAP stochasticity (MNIST experiments)

UMAP dimensionality reduction introduces randomness.  The `data_gen.py` uses
`random_state=42` for UMAP, so the embeddings are deterministic given a fixed
`DIMENSIONS` value.  However, if `DIMENSIONS` changes, the UMAP structure changes
non-trivially — it is not a simple projection.  This means Exp 6 and Exp 8 are
not directly comparable across `dimensions` values; they test different
representations, not just different capacities.  This should be stated clearly.

### 6.5 Two-layer architecture ceiling

The HNN uses exactly two `HypLinear` layers.  It is possible that a deeper
hyperbolic network would show a stronger advantage (or that the two-layer
architecture is already sufficient to demonstrate the effect).  This is a scope
limitation of the current study.

---

## 7. Paper Narrative Map

The experiments are designed to support the following paper narrative:

1. **Introduction:** Hyperbolic geometry matches tree structure (Gromov, Nickel &
   Kiela 2017).  We ask: does this translate to measurable advantages for neural
   network classification and regression?

2. **Exp 1 (anchor result):** Yes — on clean hierarchical data, hyperbolic+Radam
   outperforms Euclidean by X pp (p < 0.05, d > 0.5).

3. **Exp 7 (optimizer necessity):** The advantage requires Riemannian optimization;
   hyperbolic+Adam lags by Y pp, confirming that manifold-aware gradient updates
   are essential, not just beneficial.

4. **Exp 2 + 5b (capacity efficiency):** Hyperbolic models achieve comparable
   accuracy with ~4× fewer hidden units on both classification (Exp 2) and
   regression (Exp 5b) — direct replication of Sala et al. (2018).

5. **Exp 4 (depth scaling):** The advantage grows with hierarchy depth (prefix 10
   < 30 < 50), providing a *quantitative* measure of "how hierarchical" data must
   be for hyperbolic geometry to help.

6. **Exp 3 (noise robustness):** The advantage erodes as noise corrupts the
   hierarchical signal; the crossover noise level quantifies the robustness of
   the hyperbolic inductive bias.

7. **Exp 5 (real data generalization):** The advantage holds on real (Mircea
   phylogenetic) tree-structured data with a regression objective, confirming
   generality beyond synthetic classification.

8. **Exp 6 + 8 (flat data control):** On MNIST (no hierarchy), the two geometries
   are statistically indistinguishable — ruling out that any observed advantage
   is a general hyperbolic model quality artifact.

9. **Conclusion:** Hyperbolic geometry is advantageous when data has explicit
   hierarchical structure, the hierarchy is deep enough, noise is low, and a
   Riemannian optimizer is used.  These conditions are characterised quantitatively
   by Experiments 3–4.

---

## 8. Checklist Before Running Each Experiment

- [ ] Data files exist for all `replace` / `dimensions` values in the sweep.
- [ ] `uv run python main.py --config <file> --dry-run` shows the expected number
      of experiments.
- [ ] W&B project name is correctly set in the YAML.
- [ ] `runs` count provides sufficient power (>= 8 for secondary experiments,
      >= 10 for primary comparisons).
- [ ] Seeds are properly varied across runs (see §6.3).
- [ ] No blocking code changes are needed (Exp 9 requires `encode()` first).

---

## 9. Reference Summary

- Ganea et al. (2018). "Hyperbolic Neural Networks." NeurIPS 2018.
- Nickel & Kiela (2017). "Poincaré Embeddings for Learning Hierarchical
  Representations." NeurIPS 2017.
- Nickel & Kiela (2018). "Learning Continuous Hierarchies in the Lorentz Model
  of Hyperbolic Geometry." ICML 2018.
- Sala et al. (2018). "Representation Tradeoffs for Hyperbolic Embeddings."
  ICML 2018.
- Sarkar (2011). "Low Distortion Delaunay Embedding of Trees in Hyperbolic
  Plane." Graph Drawing 2011.
- Bonnabel (2013). "Stochastic Gradient Descent on Riemannian Manifolds."
  IEEE TAC 2013.
- Krioukov et al. (2010). "Hyperbolic Geometry of Complex Networks." Phys.
  Rev. E 2010.
- Chami et al. (2019). "Hyperbolic Graph Convolutional Neural Networks."
  NeurIPS 2019.
- Gu et al. (2019). "Learning Mixed-Curvature Representations in Product
  Spaces." ICLR 2019.
- Ermolov et al. (2022). "Hyperbolic Vision Transformers: Combining Improvements
  in Metric Learning." CVPR 2022.
- McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection."
  arXiv 2018.
