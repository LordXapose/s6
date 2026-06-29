# Sub-task 6 Multiclass SVM: One-vs-One vs One-vs-Rest

> ADT · Assignment S5 (Support Vector Machines and Kernels) · final sub-task
> A Support Vector Machine is fundamentally a **binary** classifier. This sub-task shows how it is wrapped to handle three classes, why the two wrapping strategies behave differently, and why on an **imbalanced intrusion problem** plain accuracy is the wrong thing to optimise.

---

## TL;DR results

| What | Result |
|------|--------|
| OvO `decision_function` columns | **3** (formula `k(k−1)/2` = 3 at k=3) |
| OvR `decision_function` columns | **3** (formula `k` = 3 at k=3) |
| OvO CV accuracy (5-fold) | **0.8650 ± 0.026** |
| OvR CV accuracy (5-fold) | **0.8633 ± 0.024** |
| Class 0 (benign) recall | 0.942 |
| Class 1 (recon) recall | 0.912 |
| **Class 2 (exploit) recall** | **0.615** ← the problem |
| Overall test accuracy | 0.900 (misleadingly healthy) |

**One-line takeaway:** accuracy looks fine at 0.90, but the rare exploit class — the one you can least afford to miss — is caught only 62% of the time. Per-class recall, not accuracy, is the metric that matters here.

---

## The data

```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=600, n_features=8, n_informative=5, n_redundant=0,
    n_classes=3, n_clusters_per_class=1, weights=[0.47, 0.45, 0.08],
    class_sep=0.7, flip_y=0.05, random_state=7)
```

A synthetic 3-class network-intrusion problem, **deliberately imbalanced**:

| Class | Meaning | Share | Count (of 600) |
|-------|---------|-------|----------------|
| 0 | benign traffic | 47% | 276 |
| 1 | reconnaissance | 45% | 270 |
| 2 | **genuine exploit** | 8% | 54 |

The 8% exploit class is the realistic part: in real traffic, the thing you most need to detect is also the rarest.

---

## Concept 1 Why an SVM needs a wrapper at all

A standard SVM finds **one** separating hyperplane between **two** classes (the widest-street boundary from sub-task 1). It has no native notion of "three classes." To classify *k* classes you must decompose the problem into a set of binary problems and combine their answers. There are two standard decompositions.

### One-vs-One (OvO)

Train one classifier for **every pair** of classes. Each classifier only ever sees the two classes it is responsible for; all other rows are ignored during its training. At prediction time every classifier casts a vote for one of its two classes, and the class with the most votes wins.

- Number of classifiers: **`k(k−1)/2`**
- At k = 3 → `3·2/2` = **3** classifiers: (0 vs 1), (0 vs 2), (1 vs 2)
- Each sub-problem is **balanced and small** (only two classes' worth of data).

### One-vs-Rest (OvR, also "one-vs-all")

Train one classifier **per class**, each asking "this class vs everything else." The class whose classifier is most confident wins.

- Number of classifiers: **`k`**
- At k = 3 → **3** classifiers: (0 vs {1,2}), (1 vs {0,2}), (2 vs {0,1})
- Each sub-problem is **lopsided**: the "rest" side lumps every other class together, so the minority class is pitted against a much larger combined negative set.

### The crossover at k = 3

```
       OvO = k(k−1)/2        OvR = k
k=3 →      3            =        3      ← they coincide here
k=4 →      6            ≠        4      ← they diverge from here on
k=5 →     10            ≠        5
```

This is exactly why the sub-task pins k = 3: it is the one value where the two formulas give the **same** count, so you can see the formulas agree before they pull apart. The cost difference (OvO grows quadratically, OvR linearly) only shows up at four classes or more.

---

## Concept 2 `decision_function` and the silent scikit-learn detail

`decision_function(X)` returns the raw, pre-threshold confidence scores — one column per internal binary decision. Reading its column count tells you how the model is wired:

```python
SVC(kernel="rbf", decision_function_shape="ovo").decision_function(X).shape[1]  # → 3
SVC(kernel="rbf", decision_function_shape="ovr").decision_function(X).shape[1]  # → 3
```

**The catch that trips people up:** scikit-learn's `SVC` **always trains One-vs-One internally**, regardless of what you pass. The `decision_function_shape` argument only *reshapes the returned scores* (collapsing the OvO votes into OvR-style per-class scores). It does **not** change which classifiers are trained.

So to get a *genuinely* OvR-trained model — one that really does fit "class vs rest" sub-problems — you must wrap it explicitly:

```python
from sklearn.multiclass import OneVsRestClassifier
OneVsRestClassifier(SVC(kernel="rbf"))
```

That explicit wrapper is why the accuracy comparison below uses `OneVsRestClassifier` and not just `decision_function_shape="ovr"`.

---

## Concept 3 Why scaling lives inside a Pipeline

An RBF SVM is **distance-based**: its kernel measures how close points are, so a feature on a large numeric scale would dominate the distance and drown out the others. Hence `StandardScaler`.

But the scaler must be fit **inside** the cross-validation loop, never on the whole table first:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
```

If you scale the full dataset before splitting, the scaler "sees" the validation rows (their mean and variance leak into training). Putting the scaler in a `Pipeline` means it refits on each fold's *training* rows only — the leak-free discipline carried over from S3. This is the same reason `cv_accuracy` in the harness expects a Pipeline if you scale.

---

## Concept 4 Why per-class metrics, not accuracy

**Accuracy** is the fraction of all predictions that are correct. On imbalanced data it is dominated by the majority classes. Here classes 0 and 1 are 95% of the data, so a model can score 0.90 while quietly failing on the 8% that matters.

The honest view is the **per-class confusion matrix** (rows = true, columns = predicted):

```
              pred 0   pred 1   pred 2
true 0  →       65       4        0
true 1  →        6      62        0
true 2  →        3       2        8     ← 5 of 13 real exploits misfiled
```

From it we derive two metrics per class:

- **Precision** = of everything the model *called* class C, how much really was C? → measures false alarms.
- **Recall** = of all the *real* class-C cases, how many did the model catch? → measures misses.

| Class | Precision | Recall | F1 | Read |
|-------|-----------|--------|----|------|
| 0 benign | 0.878 | 0.942 | 0.909 | solid |
| 1 recon | 0.912 | 0.912 | 0.912 | solid |
| 2 exploit | **1.000** | **0.615** | 0.762 | never over-called, often **missed** |

Class 2 has **perfect precision but poor recall**: when the model says "exploit" it is always right, but it stays silent on nearly 4 in 10 real exploits, filing them as benign or recon instead.

---

## Concept 5 The cost-aware choice

In an intrusion-detection setting the two error types are not equally expensive:

- **False positive on class 2** (benign flagged as exploit) → an analyst wastes a few minutes reviewing. Cheap.
- **False negative on class 2** (real exploit filed as benign) → the attack goes undetected. **Expensive.**

So the metric to optimise is **class-2 recall**, and the headline accuracy of 0.90 is actively misleading because it is propped up by the two large, easy classes.

**Wrapper decision:** OvO (0.8650) and OvR (0.8633) are a statistical tie, so accuracy alone does not choose between them. I keep the default **OvO `SVC`**: identical accuracy at this k, and OvO trains on *balanced* binary sub-problems rather than pitting the tiny exploit class against the combined mass of everything else (the lopsided situation OvR creates), which is gentler on the minority recall that matters most.

**To actually fix class-2 recall** (beyond the scope of the bare comparison, but the right next step):
- `class_weight="balanced"` — penalises misclassifying the rare class more heavily.
- Lower class 2's decision threshold — trade some of its perfect precision for more catches.

Both accept slightly more false alarms in exchange for missing fewer real exploits — the correct trade when a miss is the costly error.

---

## How to reproduce

```bash
pip install scikit-learn numpy matplotlib
python subtask6.py
```

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

X, y = make_classification(
    n_samples=600, n_features=8, n_informative=5, n_redundant=0,
    n_classes=3, n_clusters_per_class=1, weights=[0.47, 0.45, 0.08],
    class_sep=0.7, flip_y=0.05, random_state=7)

# 1. decision_function column counts + formula check
ovo = make_pipeline(StandardScaler(), SVC(kernel="rbf", decision_function_shape="ovo")).fit(X, y)
ovr = make_pipeline(StandardScaler(), SVC(kernel="rbf", decision_function_shape="ovr")).fit(X, y)
print("OvO cols:", ovo.decision_function(X).shape[1], " k(k-1)/2 =", 3*2//2)
print("OvR cols:", ovr.decision_function(X).shape[1], " k =", 3)

# 2. genuine OvO vs OvR cross-validated accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ovo_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf"))                       # SVC is OvO internally
ovr_clf = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(kernel="rbf"))) # genuine OvR
print("OvO CV:", cross_val_score(ovo_clf, X, y, cv=cv).mean())
print("OvR CV:", cross_val_score(ovr_clf, X, y, cv=cv).mean())

# 3. per-class confusion matrix on held-out split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(Xtr, ytr)
ypred = model.predict(Xte)
print(confusion_matrix(yte, ypred))
print(precision_recall_fscore_support(yte, ypred, labels=[0, 1, 2]))
```

All numbers reproduce exactly from the seeds shown (`random_state=7` for the data, `random_state=42` for the split and CV splitter).

---

## Glossary

| Term | Meaning |
|------|---------|
| **OvO** | One-vs-One: one classifier per *pair* of classes, `k(k−1)/2` total, majority vote. |
| **OvR** | One-vs-Rest: one classifier per class ("class vs everything"), `k` total. |
| **`decision_function`** | Raw pre-threshold confidence scores; column count reveals the wiring. |
| **Precision** | Of predicted-C, how many were really C (false-alarm view). |
| **Recall** | Of real C, how many were caught (miss view). |
| **Stratified split/CV** | Keeps each class's proportion in every fold — essential so rare class 2 appears in the test fold. |
| **Leakage** | Letting validation data influence training (e.g. scaling before splitting); avoided by the Pipeline. |
