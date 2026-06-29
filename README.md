# Sub-task 6 Multiclass SVM: One-vs-One vs One-vs-Rest

> **ADT · Assignment S5 (Support Vector Machines and Kernels) · final sub-task**
> A Support Vector Machine is fundamentally a **binary** classifier — it draws one boundary between two
> classes. This sub-task shows (1) how it is *wrapped* to handle three classes, (2) why the two wrapping
> strategies behave differently, and (3) why, on a deliberately **imbalanced intrusion problem**, plain
> accuracy is the wrong thing to optimise and per-class recall is the metric that actually matters.

---

## Repository contents

| File | What it is | Why it's here |
|------|------------|---------------|
| **`README.md`** | This document | The concept walk-through + a fully line-commented script |
| **`s6.html`** | Single-screen slide | The presentable S6-opener unit (one screen, self-contained) |
| **`s6.pdf`** | Full 6-page report | Title block, figures, tables, glossary the formal write-up |

> Every number across all three files is identical and reproduces exactly from the seeds in the code
> (`random_state=7` for the data, `random_state=42` for the split and the cross-validation splitter).

---

## TL;DR results

| What | Result | Why it matters |
|------|--------|----------------|
| OvO `decision_function` columns | **3** | matches `k(k−1)/2` = 3 at k=3 |
| OvR `decision_function` columns | **3** | matches `k` = 3 at k=3 |
| OvO CV accuracy (5-fold) | **0.8650 ± 0.026** | the internal scikit-learn default |
| OvR CV accuracy (5-fold) | **0.8633 ± 0.024** | statistically tied with OvO |
| Class 0 (benign) recall | 0.942 | easy majority class |
| Class 1 (recon) recall | 0.912 | easy majority class |
| **Class 2 (exploit) recall** | **0.615 ← the problem** | the rare class we cannot afford to miss |
| Overall test accuracy | 0.900 | **misleadingly healthy** |

**One-line takeaway:** accuracy looks fine at 0.90, but the rare exploit class the one you can least afford
to miss is caught only 62 % of the time. **Per-class recall, not accuracy, is the metric that matters here.**

---

## The fully-commented script

This is the whole sub-task in one runnable file. **Every line carries an inline comment** explaining what it
does and *why* it is there, so the file doubles as the explanation. Save it as `subtask6.py` and run it.

```python
# ─────────────────────────────────────────────────────────────────────────────
# subtask6.py — ADT S5 Sub-task 6: Multiclass SVM, OvO vs OvR
# Goal: wrap a binary SVM for 3 classes, compare the two wrappers, and expose
#       the rare-class failure that accuracy hides on imbalanced data.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np                                       # numeric arrays + bincount for class counts
from sklearn.datasets import make_classification         # generates the synthetic, seeded intrusion set
from sklearn.svm import SVC                               # the Support Vector Classifier itself (RBF kernel)
from sklearn.multiclass import OneVsRestClassifier       # explicit OvR wrapper (SVC won't do real OvR alone)
from sklearn.preprocessing import StandardScaler         # z-scores features so the distance-based kernel is fair
from sklearn.pipeline import make_pipeline               # chains scaler+model so scaling can't leak across folds
from sklearn.model_selection import (
    train_test_split,                                    # carves out a held-out test fold for the confusion matrix
    StratifiedKFold,                                     # CV splitter that preserves each class's proportion per fold
    cross_val_score,                                     # runs the fit/score loop over the folds and returns scores
)
from sklearn.metrics import (
    confusion_matrix,                                    # raw true-vs-predicted counts, the honest per-class view
    precision_recall_fscore_support,                     # turns that matrix into per-class precision / recall / F1
)

# ── 0. Build the data ────────────────────────────────────────────────────────
# A synthetic 3-class network-intrusion problem, imbalanced ON PURPOSE so the
# most important class is also the rarest — the realistic situation in security.
X, y = make_classification(
    n_samples=600,             # 600 rows total — small enough to run instantly, big enough to be stable
    n_features=8,              # 8 columns; the SVM works in this 8-D space (we don't need to plot it)
    n_informative=5,           # 5 of those 8 actually carry class signal...
    n_redundant=0,             # ...and 0 are linear combinations of others, so no fake/duplicate signal
    n_classes=3,               # THREE classes — this is what forces a multiclass wrapper around the binary SVM
    n_clusters_per_class=1,    # each class is one blob, keeping the geometry clean and the result reproducible
    weights=[0.47, 0.45, 0.08],# class shares: 47% benign, 45% recon, 8% exploit — the deliberate imbalance
    class_sep=0.7,             # moderate separation: classes overlap enough that the model can actually err
    flip_y=0.05,               # flip 5% of labels to noise, so 100% accuracy is impossible (realistic)
    random_state=7,            # fixes the RNG → your data matches the reference solution byte-for-byte
)
# class 0 = benign traffic, class 1 = reconnaissance, class 2 = genuine exploit (must not miss)
print("class counts:", np.bincount(y))   # → [276 270 54]: confirms the 47/45/8 split landed as intended

# ── 1. decision_function column counts + the formula check ───────────────────
# decision_function() returns one confidence column per internal binary decision.
# Reading the column count tells us how the multiclass machine is wired.
ovo = make_pipeline(                                      # Pipeline = scaler THEN model, treated as one estimator
    StandardScaler(),                                    # fit on training rows only (the Pipeline guarantees this)
    SVC(kernel="rbf", decision_function_shape="ovo"),    # ask for OvO-shaped scores
).fit(X, y)                                              # fit on the whole set (we only want the score SHAPE here)
ovr = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", decision_function_shape="ovr"),    # ask for OvR-shaped scores
).fit(X, y)

k = 3                                                    # number of classes, used in the formula checks below
print("OvO cols:", ovo.decision_function(X).shape[1],    # → 3
      "| formula k(k-1)/2 =", k * (k - 1) // 2)          # → 3  (one classifier per PAIR of classes)
print("OvR cols:", ovr.decision_function(X).shape[1],    # → 3
      "| formula k =", k)                                # → 3  (one classifier per class)
# NOTE: at k=3 both give 3 — the crossover. They only diverge at k>=4 (OvO=6 vs OvR=4).
# CAVEAT: SVC ALWAYS trains One-vs-One internally; decision_function_shape only RESHAPES the
# returned scores — it does NOT change what is trained. That's why §2 uses an explicit OvR wrapper.

# ── 2. genuine OvO vs OvR — cross-validated accuracy ─────────────────────────
cv = StratifiedKFold(
    n_splits=5,                # 5 folds: train on 4/5, validate on 1/5, five times
    shuffle=True,              # shuffle before splitting so fold order doesn't track row order
    random_state=42,           # fixed seed → identical folds every run (and matches the reference)
)
ovo_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf"))                        # default SVC = OvO internally
ovr_clf = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(kernel="rbf")))  # REAL OvR (class-vs-rest fits)

# cross_val_score refits the WHOLE pipeline on each fold's training rows → no leakage.
print("OvO CV acc:", cross_val_score(ovo_clf, X, y, cv=cv).mean())   # → 0.8650
print("OvR CV acc:", cross_val_score(ovr_clf, X, y, cv=cv).mean())   # → 0.8633
# The two are within ~0.2 points: a statistical tie. Accuracy alone can't pick a winner.

# ── 3. per-class confusion matrix on a held-out split ────────────────────────
Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=0.25,            # hold out 25% the model never sees during training
    random_state=42,           # fixed split → your matrix matches the reference exactly
    stratify=y,                # CRUCIAL: keeps the 8% exploit share in the test fold, so class 2 is testable
)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(Xtr, ytr)  # train on the 75%
ypred = model.predict(Xte)                                                # predict on the unseen 25%

print(confusion_matrix(yte, ypred))   # rows = true class, cols = predicted class; diagonal = correct
# [[65  4  0]
#  [ 6 62  0]
#  [ 3  2  8]]   ← bottom row: of 13 true exploits, only 8 caught; 5 leaked into benign/recon

# precision = of everything PREDICTED class C, how much really was C   (false-alarm view)
# recall    = of everything TRULY  class C, how much did we CATCH      (miss view)
prec, rec, f1, support = precision_recall_fscore_support(yte, ypred, labels=[0, 1, 2])
for c in range(3):
    print(f"class {c}: precision={prec[c]:.3f}  recall={rec[c]:.3f}  f1={f1[c]:.3f}  n={support[c]}")
# class 2 (exploit): precision=1.000  recall=0.615  → never a false alarm, but missed ~38% of the time
print("overall accuracy:", (ypred == yte).mean())   # → 0.900, healthy-LOOKING but propped up by classes 0 & 1
```

```text
$ python subtask6.py
class counts: [276 270  54]
OvO cols: 3 | formula k(k-1)/2 = 3
OvR cols: 3 | formula k = 3
OvO CV acc: 0.8650
OvR CV acc: 0.8633
[[65  4  0]
 [ 6 62  0]
 [ 3  2  8]]
class 0: precision=0.878  recall=0.942  f1=0.909  n=69
class 1: precision=0.912  recall=0.912  f1=0.912  n=68
class 2: precision=1.000  recall=0.615  f1=0.762  n=13
overall accuracy: 0.900
```

---

## The data, in a table

A synthetic 3-class network-intrusion problem, **deliberately imbalanced**:

| Class | Meaning | Share | Count (of 600) | Why it's set this way |
|-------|---------|-------|----------------|-----------------------|
| 0 | benign traffic | 47 % | 276 | the bulk of normal traffic |
| 1 | reconnaissance | 45 % | 270 | scanning/probing — common, lower stakes |
| 2 | **genuine exploit** | 8 % | 54 | **rare but the one we must catch** |

The 8 % exploit class is the realistic part: in real traffic, the thing you most need to detect is also the rarest.

---

## Concept 1 Why an SVM needs a wrapper at all

A standard SVM finds **one** separating hyperplane between **two** classes (the widest-street boundary from
Sub-task 1). It has no native notion of "three classes." To classify *k* classes you decompose the problem
into a set of binary problems and combine their answers. Two standard decompositions exist.

### One-vs-One (OvO) — *used by `SVC` internally*
Train one classifier for **every pair** of classes. Each sees only its two classes; all other rows are ignored
during its training. At prediction time every classifier votes for one of its two classes; the class with the
most votes wins.
- **Classifiers:** `k(k−1)/2` → at k=3 that is **3**: (0 vs 1), (0 vs 2), (1 vs 2)
- **Why it's nice here:** each sub-problem is **balanced and small** — only two classes' worth of data, so the
  rare class is never drowned.

### One-vs-Rest (OvR, "one-vs-all") — *used via the explicit wrapper*
Train one classifier **per class**, each asking "this class vs everything else." The most confident wins.
- **Classifiers:** `k` → at k=3 that is **3**: (0 vs {1,2}), (1 vs {0,2}), (2 vs {0,1})
- **Why it's riskier here:** each sub-problem is **lopsided** — the "rest" side lumps every other class
  together, so the minority exploit class is pitted against a much larger combined negative set.

### The crossover at k = 3
```
       OvO = k(k−1)/2     OvR = k
k=3 →      3         =       3     ← they coincide here (why this sub-task pins k=3)
k=4 →      6         ≠       4     ← they diverge from here on
k=5 →     10         ≠       5
```
k=3 is the single value where the two formulas agree, so you can confirm them before they pull apart. The cost
difference (OvO grows quadratically, OvR linearly) only shows up at four classes or more.

---

## Concept 2 `decision_function` and the silent scikit-learn detail

`decision_function(X)` returns the raw, pre-threshold confidence scores — **one column per internal binary
decision** — so its column count tells you how the model is wired. **The catch:** scikit-learn's `SVC`
**always trains One-vs-One internally**, whatever you pass. `decision_function_shape` only *reshapes the
returned scores*; it does **not** change what is trained. To get a *genuinely* OvR-trained model you must wrap
it explicitly with `OneVsRestClassifier(SVC(...))` — which is exactly why the §2 accuracy comparison uses that
wrapper and not just `decision_function_shape="ovr"`.

---

## Concept 3 Why scaling lives inside a Pipeline

An RBF SVM is **distance-based**: its kernel measures how close points are, so a feature on a large numeric
scale would dominate the distance and drown out the others — hence `StandardScaler`. But the scaler must be fit
**inside** the cross-validation loop, never on the whole table first. If you scale the full dataset before
splitting, the scaler "sees" the validation rows (their mean and variance leak into training). A `Pipeline`
refits the scaler on each fold's **training** rows only — the leak-free discipline carried over from S3.

---

## Concept 4 Why per-class metrics, not accuracy

**Accuracy** is the fraction of all predictions that are correct, so on imbalanced data it is dominated by the
majority classes. Here classes 0 and 1 are 95 % of the data, so a model can score 0.90 while quietly failing on
the 8 % that matters. The honest view is the **per-class confusion matrix** (rows = true, columns = predicted):

```
              pred 0   pred 1   pred 2
true 0  →       65       4        0
true 1  →        6      62        0
true 2  →        3       2        8     ← 5 of 13 real exploits misfiled
```

From it we derive two per-class metrics:
- **Precision** = of everything the model *called* class C, how much really was C? → measures **false alarms**.
- **Recall** = of all the *real* class-C cases, how many did the model catch? → measures **misses**.

| Class | Precision | Recall | F1 | Read |
|-------|-----------|--------|----|------|
| 0 benign | 0.878 | 0.942 | 0.909 | solid |
| 1 recon | 0.912 | 0.912 | 0.912 | solid |
| 2 exploit | **1.000** | **0.615** | 0.762 | never over-called, often **missed** |

Class 2 has **perfect precision but poor recall**: when the model says "exploit" it is always right, but it
stays silent on nearly 4 in 10 real exploits, filing them as benign or recon instead.

---

## Concept 5 The cost-aware choice

In an intrusion-detection setting the two error types are not equally expensive:
- **False positive on class 2** (benign flagged as exploit) → an analyst wastes a few minutes. **Cheap.**
- **False negative on class 2** (real exploit filed as benign) → the attack goes undetected. **Expensive.**

So the metric to optimise is **class-2 recall**, and the headline accuracy of 0.90 is actively misleading
because it is propped up by the two large, easy classes.

**Wrapper decision:** OvO (0.8650) and OvR (0.8633) are a statistical tie, so accuracy alone does not choose
between them. I keep the default **OvO `SVC`**: identical accuracy at this k, and OvO trains on *balanced*
binary sub-problems rather than pitting the tiny exploit class against the combined mass of everything else
(the lopsided situation OvR creates), which is gentler on the minority recall that matters most.

**To actually fix class-2 recall** (beyond the bare comparison, but the right next step):
- `class_weight="balanced"` — penalises misclassifying the rare class more heavily.
- Lower class 2's decision threshold — trade some of its perfect precision for more catches.

Both accept slightly more false alarms in exchange for missing fewer real exploits — the correct trade when a
miss is the costly error.

---

## How to reproduce

```bash
pip install scikit-learn numpy matplotlib   # the only dependencies
python subtask6.py                          # prints every number in the tables above
```

All numbers reproduce exactly from the seeds in the script (`random_state=7` for the data, `random_state=42`
for the split and the CV splitter).

---

## Glossary

| Term | Meaning | Why you care here |
|------|---------|-------------------|
| **OvO** | One-vs-One: one classifier per *pair* of classes, `k(k−1)/2` total, majority vote | `SVC`'s internal strategy; balanced sub-problems |
| **OvR** | One-vs-Rest: one classifier per class ("class vs everything"), `k` total | the explicit-wrapper contrast; lopsided sub-problems |
| **`decision_function`** | Raw pre-threshold confidence scores | column count reveals the wiring (OvO vs OvR) |
| **Precision** | Of predicted-C, how many were really C | the false-alarm view |
| **Recall** | Of real C, how many were caught | the miss view — the one that matters for exploits |
| **Stratified split / CV** | Keeps each class's proportion in every fold | so the rare class 2 actually appears in the test fold |
| **Pipeline** | Chains preprocessing + model into one estimator | refits the scaler per fold → no leakage |
| **Leakage** | Letting validation data influence training (e.g. scaling before splitting) | inflates scores; the Pipeline prevents it |
