V1 baseline — TF-IDF + Logistic Regression

Macro-F1: 0.9588
Accuracy: 0.9600
Training: 1.338 s
p50 latency: 0.370 ms
p95 latency: 0.464 ms

Observation:
A simple supervised text classifier performs extremely well on the
stable V1 taxonomy. This establishes a strong classical baseline:
Jev or an SLM must justify itself through adaptability or other
operational advantages, not merely classification accuracy.

## V1 Baseline Results

### Experiment question

Before testing Jev's adaptability, establish how well conventional supervised classifiers perform when the business taxonomy is stable and sufficient labelled data already exists.

The V1 taxonomy contains seven business queues.

The same frozen 600-example test set is used for all models. Macro-F1 is the primary quality metric because the class distribution is uneven; accuracy is reported as a secondary overall measure.

---

### Baseline 1 — TF-IDF + Logistic Regression

Pipeline:

```text
customer text
    ↓
TF-IDF unigram + bigram features
    ↓
Logistic Regression
    ↓
V1 queue
```

Configuration:

- TF-IDF with unigrams and bigrams
- `min_df=2`
- sublinear term frequency
- Logistic Regression
- 10,003 BANKING77 training examples
- 600 frozen test examples
- 7 V1 classes

Results:

| Metric            |        Result |
| ----------------- | ------------: |
| Macro-F1          |        0.9588 |
| Accuracy          |        0.9600 |
| Training time     |       1.338 s |
| Batch prediction  | 0.008 s / 600 |
| Mean latency      |      0.383 ms |
| p50 latency       |      0.370 ms |
| p95 latency       |      0.464 ms |
| TF-IDF vocabulary |        10,292 |

Per-class F1:

| V1 queue        |     F1 |
| --------------- | -----: |
| ACCOUNT_ACCESS  | 0.9767 |
| CARD_PAYMENTS   | 0.9434 |
| CARD_SERVICES   | 0.9679 |
| CASH_WITHDRAWAL | 0.9462 |
| CURRENCY        | 0.9778 |
| TOP_UPS         | 0.9613 |
| TRANSFERS       | 0.9383 |

Observation:

The simple classical classifier performs extremely well on the stable V1 taxonomy. Macro-F1 and accuracy are almost identical, indicating that the high overall score is not being produced only by the larger classes.

The result establishes a strong baseline: Jev or an SLM cannot justify itself merely by being able to classify banking requests. Any advantage will likely need to come from adaptability, reduced labelling requirements, or other operational properties.

---

### Baseline 2 — MiniLM Embeddings + Logistic Regression

Pipeline:

```text
customer text
    ↓
all-MiniLM-L6-v2
    ↓
384-dimensional sentence embedding
    ↓
Logistic Regression
    ↓
V1 queue
```

Embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

Results:

| Metric                       |        Result |
| ---------------------------- | ------------: |
| Macro-F1                     |        0.9508 |
| Accuracy                     |        0.9567 |
| Training embedding time      |      23.236 s |
| Test embedding time          |       1.847 s |
| Logistic Regression training |       2.122 s |
| Classifier batch prediction  | 0.001 s / 600 |
| Mean end-to-end latency      |     13.452 ms |
| p50 latency                  |     13.338 ms |
| p95 latency                  |     17.866 ms |
| Embedding dimension          |           384 |

Per-class F1:

| V1 queue        |     F1 |
| --------------- | -----: |
| ACCOUNT_ACCESS  | 0.9767 |
| CARD_PAYMENTS   | 0.9554 |
| CARD_SERVICES   | 0.9736 |
| CASH_WITHDRAWAL | 0.9053 |
| CURRENCY        | 0.9565 |
| TOP_UPS         | 0.9560 |
| TRANSFERS       | 0.9317 |

---

### V1 comparison

| Model                        | Macro-F1 | Accuracy | p50 latency | p95 latency |
| ---------------------------- | -------: | -------: | ----------: | ----------: |
| TF-IDF + Logistic Regression |   0.9588 |   0.9600 |    0.370 ms |    0.464 ms |
| MiniLM + Logistic Regression |   0.9508 |   0.9567 |   13.338 ms |   17.866 ms |

On this dataset, the embedding classifier did not improve classification quality.

Its p50 end-to-end inference latency was approximately:

```text
13.338 / 0.370 ≈ 36×
```

that of TF-IDF + Logistic Regression on the test machine.

This does not imply that embedding classifiers are generally inferior. BANKING77 contains short, relatively explicit customer queries with strong lexical cues such as "withdrawal", "transfer", "card payment", and "currency". TF-IDF can exploit these cues very effectively.

The embedding representation provides richer semantic information, but that additional representational capacity did not translate into better V1 performance in this experiment.

---

### Initial interpretation

The V1 results make the next experiment more important.

For a stable business taxonomy with sufficient labelled data:

```text
TF-IDF + Logistic Regression
    → high accuracy
    → sub-millisecond inference
    → negligible local operating cost
    → very simple implementation
```

The core question is therefore no longer:

> Can Jev classify the requests?

Instead:

> What happens when the business taxonomy changes and the existing supervised classifier no longer matches the decision the business wants to make?

The V2 experiment will test whether promptable decision models such as Jev or an SLM can adapt immediately from new class definitions, while conventional ML requires new V2-labelled examples and retraining.

---

### Implementation notes

The MiniLM model was downloaded from Hugging Face on the first run.

Two warnings appeared:

- unauthenticated Hugging Face requests, affecting download rate limits only;
- Windows does not support Hugging Face cache symlinks by default, resulting in potentially higher local disk usage.

Neither warning affects the benchmark results.

Training and test embeddings were cached locally so subsequent runs do not need to recompute them.


## V2 Adaptation Results — Supervised ML

The V2 experiment simulates a business taxonomy change.

Historical data from clean V1 queues was reused:

- TRANSFERS
- TOP_UPS
- CURRENCY

Historical examples from impure V1 queues were not deterministically reusable because their old labels mapped into multiple V2 queues.

For the six affected V2 classes, two new-label budgets were tested:

- 20 labels/class = 120 new V2 labels total
- 100 labels/class = 600 new V2 labels total

Each condition used five independently sampled draws. Logistic Regression used balanced class weights because the V2 training set combines thousands of reusable historical examples with very small newly labelled classes.

### Aggregated results

| Model          | New labels | Macro-F1 mean | Macro-F1 range | Accuracy mean |
| -------------- | ---------: | ------------: | -------------: | ------------: |
| TF-IDF + LR    |        120 |        0.6810 |  0.6549–0.7114 |        0.6843 |
| Embedding + LR |        120 |        0.8460 |  0.8261–0.8633 |        0.8450 |
| TF-IDF + LR    |        600 |        0.8864 |  0.8795–0.9003 |        0.8837 |
| Embedding + LR |        600 |        0.9065 |  0.8958–0.9172 |        0.9040 |

### Observation

The embedding classifier showed a substantial advantage in the low-label adaptation regime.

With only 20 new examples per affected V2 class:

```text
TF-IDF + LR    Macro-F1 = 0.6810
Embedding + LR Macro-F1 = 0.8460
```

The gap was 0.165 Macro-F1.

At 100 labels/class, the gap narrowed considerably:

```text
TF-IDF + LR    Macro-F1 = 0.8864
Embedding + LR Macro-F1 = 0.9065
```

This suggests that semantic embeddings provide their largest value when the classification boundary changes but relatively few newly labelled examples are available.

As more V2-labelled data becomes available, the simpler lexical classifier catches up.

This contrasts with the V1 result, where TF-IDF slightly outperformed the embedding classifier.

The emerging interpretation is therefore not that embeddings are universally superior, but that they are more label-efficient during taxonomy adaptation.

## Zero-Shot SLM Baseline

A Qwen3-4B-Instruct-2507 Q4_K_M model was tested against the V2 taxonomy using Ollama.

The model received:

- the raw customer message;
- the nine frozen V2 class definitions;
- no BANKING77 intent;
- no V1 label;
- no V2 training examples.

The model therefore used **zero new V2 labels**.

### Results

| Metric          |     Result |
| --------------- | ---------: |
| Macro-F1        |     0.7749 |
| Accuracy        |     0.7767 |
| Invalid outputs |    0 / 600 |
| Mean latency    | 1290.82 ms |
| p50 latency     | 1214.45 ms |
| p95 latency     | 1848.34 ms |

### Comparison with adapted ML

| Method             | New V2 labels | Macro-F1 |
| ------------------ | ------------: | -------: |
| TF-IDF + LR        |           120 |   0.6810 |
| Qwen3-4B zero-shot |             0 |   0.7749 |
| Embedding + LR     |           120 |   0.8460 |
| TF-IDF + LR        |           600 |   0.8864 |
| Embedding + LR     |           600 |   0.9065 |

The SLM outperformed TF-IDF trained with 20 labels per affected class despite using no new V2 labels. However, it remained below the embedding classifier trained with the same 20-label budget.

The result suggests that a small language model can provide immediate adaptability when the taxonomy changes, but there is a quality and latency cost compared with supervised adaptation.

### Per-class observation

The SLM performed particularly well on IDENTITY_COMPLIANCE and CARD_MANAGEMENT.

The weakest areas were DIGITAL_PAYMENTS and SECURITY_DISPUTES.

SECURITY_DISPUTES had high recall but low precision, suggesting the model tended to over-route suspicious or ambiguous requests into the security category.

This is an important distinction between semantic understanding and a well-calibrated business decision boundary.

## Zero-Shot Jev Results

### Experiment setup

Jev was evaluated against exactly the same frozen V2 benchmark used for the zero-shot SLM experiment:

- 600 frozen BANKING77 test messages
- 9 V2 business classes
- identical V2 class definitions
- no BANKING77 original intent exposed to the model
- no V1 label exposed to the model
- no V2 training examples
- no few-shot demonstrations
- zero new V2 labels

The requested API model was:

`jev-latest`

The TypeSafe API reported the actual serving model as:

`jev-1.13.0`

Unlike the SLM, Jev returned a native typed choice together with a probability distribution and a separate confidence value. No free-text output parsing was required.

---

### Jev classification results

| Metric                               |    Result |
| ------------------------------------ | --------: |
| Macro-F1                             |    0.8954 |
| Accuracy                             |    0.8900 |
| New V2 labels                        |         0 |
| Test examples                        |       600 |
| Mean latency                         | 723.01 ms |
| p50 latency                          | 684.89 ms |
| p95 latency                          | 992.44 ms |
| Input tokens                         |   325,972 |
| Total API cost                       | $0.013691 |
| Approx. cost / 1,000 classifications |   $0.0228 |

Per-class results:

| V2 class            | Precision | Recall |     F1 |
| ------------------- | --------: | -----: | -----: |
| ACCOUNT_SERVICES    |    0.8085 | 0.9500 | 0.8736 |
| CARD_MANAGEMENT     |    0.9338 | 0.9407 | 0.9373 |
| CASH_WITHDRAWAL     |    0.8696 | 1.0000 | 0.9302 |
| CURRENCY            |    0.8980 | 0.9565 | 0.9263 |
| DIGITAL_PAYMENTS    |    0.8451 | 0.7692 | 0.8054 |
| IDENTITY_COMPLIANCE |    1.0000 | 0.9750 | 0.9873 |
| SECURITY_DISPUTES   |    0.9524 | 0.8511 | 0.8989 |
| TOP_UPS             |    0.8333 | 0.8242 | 0.8287 |
| TRANSFERS           |    0.8875 | 0.8554 | 0.8712 |

No V2 class collapsed. The weakest class was DIGITAL_PAYMENTS at 0.8054 F1, while IDENTITY_COMPLIANCE reached 0.9873.

---

## V2 Comparison So Far

| Method             | New V2 labels |   Macro-F1 |   Accuracy |
| ------------------ | ------------: | ---------: | ---------: |
| TF-IDF + LR        |           120 |     0.6810 |     0.6843 |
| Qwen3-4B zero-shot |             0 |     0.7749 |     0.7767 |
| Embedding + LR     |           120 |     0.8460 |     0.8450 |
| **Jev zero-shot**  |         **0** | **0.8954** | **0.8900** |
| TF-IDF + LR        |           600 |     0.8864 |     0.8837 |
| Embedding + LR     |           600 |     0.9065 |     0.9040 |

### Main observation

Jev achieved 0.8954 Macro-F1 without receiving any newly labelled V2 examples.

This was:

- 0.1205 higher than Qwen3-4B zero-shot;
- 0.0494 higher than Embedding + LR with 120 new labels;
- 0.0090 higher than TF-IDF + LR with 600 new labels;
- 0.0111 lower than Embedding + LR with 600 new labels.

The result therefore does not suggest that Jev universally replaces supervised classifiers.

Instead, Jev appears particularly strong in the period immediately after a decision taxonomy changes, before an organisation has collected enough new labels to retrain a task-specific classifier.

A semantic embedding classifier eventually achieved slightly higher quality once 600 new V2 labels were available.

---

## Zero-Shot SLM vs Jev

Both systems received:

- the same customer messages;
- the same frozen V2 class definitions;
- zero new V2 labels.

| Model                         |   Macro-F1 |   Accuracy |   p50 latency |
| ----------------------------- | ---------: | ---------: | ------------: |
| Qwen3-4B-Instruct-2507 Q4_K_M |     0.7749 |     0.7767 |    1214.45 ms |
| Jev (`jev-1.13.0`)            | **0.8954** | **0.8900** | **684.89 ms** |

Jev exceeded the Qwen3-4B baseline by 0.1205 Macro-F1.

A particularly important difference appeared in SECURITY_DISPUTES.

Qwen3-4B:

- precision: 0.4300
- recall: 0.9149
- F1: 0.5850

Jev:

- precision: 0.9524
- recall: 0.8511
- F1: 0.8989

Qwen tended to route too many suspicious or ambiguous messages into SECURITY_DISPUTES. Jev produced a substantially more precise decision boundary.

Latency must be interpreted cautiously. Qwen was run locally on the test laptop while Jev was accessed through a hosted API. The observed end-to-end latency therefore reflects the tested deployment configurations, not an intrinsic architectural speed comparison.

---

# Jev Probability and Confidence Analysis

Jev returns two distinct uncertainty-related outputs:

1. a probability distribution across the available choices;
2. a separate API `confidence` score.

These should not be treated as the same quantity.

Across the 600 examples, the maximum observed difference between API confidence and the raw selected-class probability was 0.09.

The returned probability vectors also summed approximately, rather than exactly, to 1.0. They were therefore normalized before calculating multiclass Brier score and log loss.

---

## Probability Calibration

| Metric                          |  Result |
| ------------------------------- | ------: |
| Accuracy                        |  0.8900 |
| Mean selected-class probability |  0.9443 |
| Probability − accuracy gap      | +0.0543 |
| ECE                             |  0.0585 |
| MCE                             |  0.1345 |
| Top-label Brier score           |  0.0856 |
| Multiclass Brier score          |  0.1815 |
| Multiclass log loss             |  1.3615 |

The mean selected-class probability was 94.43%, while actual accuracy was 89.00%.

This suggests that Jev was **somewhat overconfident on this benchmark**.

The fixed-bin Expected Calibration Error was 0.0585, meaning predicted probability differed from observed accuracy by approximately 5.9 percentage points on average across the bins, weighted by the number of predictions in each bin.

The largest group contained 499 of the 600 examples:

| Selected probability band | Cases | Mean probability | Actual accuracy |
| ------------------------- | ----: | ---------------: | --------------: |
| 0.90–1.00                 |   499 |           0.9887 |          0.9399 |

For these predictions, Jev assigned an average selected-class probability of almost 99%, while observed accuracy was about 94%.

The probabilities therefore contain useful uncertainty information, but they should not be interpreted as perfectly calibrated probabilities on this dataset.

---

## Jev API Confidence as a Risk Signal

The separate Jev API confidence field was analysed as a ranking signal rather than as a literal probability of correctness.

| Confidence group | Cases | Mean confidence | Accuracy | Error rate |
| ---------------- | ----: | --------------: | -------: | ---------: |
| Lowest           |    63 |          0.5943 |   0.5873 |     0.4127 |
| 2                |    63 |          0.8643 |   0.7460 |     0.2540 |
| 3                |    67 |          0.9488 |   0.8955 |     0.1045 |
| 4                |    64 |          0.9861 |   0.8750 |     0.1250 |
| Highest          |   343 |          1.0000 |   0.9738 |     0.0262 |

The relationship is not perfectly monotonic, but the broad pattern is strong:

- low-confidence decisions have much higher error rates;
- high-confidence decisions are substantially more reliable.

The 343 cases with API confidence 1.0 achieved 97.38% accuracy.

This also demonstrates that `confidence = 1.0` should not be interpreted as a literal 100% probability of being correct.

---

## Human-Review Simulation

Jev made 66 errors among the 600 test examples.

The examples were ranked from lowest to highest API confidence to simulate sending uncertain decisions to human review.

| Lowest-confidence cases reviewed | Requests reviewed | Errors caught | Share of all errors caught |
| -------------------------------- | ----------------: | ------------: | -------------------------: |
| 10%                              |                60 |            24 |                      36.4% |
| 20%                              |               120 |            39 |                  **59.1%** |
| 30%                              |               180 |            48 |                  **72.7%** |

### Operational observation

Reviewing only the lowest-confidence 20% of Jev decisions would have surfaced 59.1% of all Jev errors in this experiment.

This may be more operationally useful than treating the confidence value as an exact probability.

It suggests a possible deployment pattern:

```text
customer request
      ↓
     Jev
      ↓
decision + confidence
      │
      ├── high confidence → automated processing
      │
      └── lower confidence → human review