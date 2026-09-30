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