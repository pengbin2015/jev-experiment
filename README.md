# Jev Experiment: Decision Models vs Traditional ML and a Small Language Model

This repository contains a practical experiment comparing four approaches to enterprise text classification:

1. **TF-IDF + Logistic Regression**
2. **Sentence Embeddings + Logistic Regression**
3. **Qwen3-4B** as a zero-shot small language model
4. **Jev** as a zero-shot specialised decision model

The experiment is designed around a question that is common in enterprise software:

> **What happens when the business taxonomy changes before we have enough newly labelled data to retrain a classifier?**

Rather than asking only which model achieves the highest classification accuracy, the experiment compares:

- classification quality;
- new-label requirements;
- adaptation to taxonomy change;
- inference latency;
- API cost;
- and, for Jev, probability calibration and confidence-based human review.

---

## Motivation

Traditional supervised classifiers can perform extremely well when:

- the taxonomy is stable;
- sufficient labelled examples already exist;
- and the business decision does not change frequently.

The more difficult situation is a **taxonomy change**.

For example, consider this customer request:

```text
"I don't recognise this ATM withdrawal."
```

Under an existing routing taxonomy, it may be classified as:

```text
CASH_WITHDRAWAL
```

After a business reorganisation, suspicious or unrecognised transactions may instead need to go to:

```text
SECURITY_DISPUTES
```

The input has not changed.

The **meaning of the business decision has changed**.

This experiment tests how different classification approaches respond to that situation.

---

# Dataset

The experiment uses the **BANKING77** dataset from PolyAI, hosted on Hugging Face:

https://huggingface.co/datasets/PolyAI/banking77

BANKING77 contains:

- **13,083** banking customer-service queries;
- **10,003** official training examples;
- **3,080** official test examples;
- **77** fine-grained intent labels.

The dataset is downloaded at a pinned Hugging Face revision for reproducibility.

The original 77 BANKING77 intents are used only to construct experiment ground truth.

They are **not exposed to the models during classification**.

---

# Experiment Design

## V1: Stable Business Taxonomy

The initial business taxonomy contains seven queues:

```text
CARD_SERVICES
CARD_PAYMENTS
CASH_WITHDRAWAL
TRANSFERS
TOP_UPS
ACCOUNT_ACCESS
CURRENCY
```

A business queue represents the team, workflow, or automated process that should receive a customer request.

For example:

```text
"My transfer has not arrived."
        ↓
TRANSFERS
```

V1 is used to establish how well traditional supervised ML performs when the taxonomy is stable and labelled data is abundant.

---

## V2: Changed Business Taxonomy

The business taxonomy is then changed to:

```text
CARD_MANAGEMENT
DIGITAL_PAYMENTS
CASH_WITHDRAWAL
TRANSFERS
TOP_UPS
ACCOUNT_SERVICES
IDENTITY_COMPLIANCE
SECURITY_DISPUTES
CURRENCY
```

Some V1 queues map cleanly into V2:

```text
TRANSFERS → TRANSFERS
TOP_UPS   → TOP_UPS
CURRENCY  → CURRENCY
```

Other V1 queues become **impure** because their historical examples now belong to several V2 classes.

For example:

```text
CARD_SERVICES
       ↓
CARD_MANAGEMENT
DIGITAL_PAYMENTS
SECURITY_DISPUTES
```

The old `CARD_SERVICES` label alone is therefore insufficient to determine the new V2 destination.

---

# Models Compared

## 1. TF-IDF + Logistic Regression

A classic supervised text-classification pipeline:

```text
Text
 ↓
TF-IDF
 ↓
Sparse lexical features
 ↓
Logistic Regression
 ↓
Business queue
```

TF-IDF represents important words and phrases numerically.

Logistic Regression learns the relationship between those features and labelled business classes.

---

## 2. Sentence Embeddings + Logistic Regression

This model uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to produce a 384-dimensional semantic embedding for each customer message.

```text
Text
 ↓
MiniLM embedding
 ↓
Semantic vector
 ↓
Logistic Regression
 ↓
Business queue
```

This remains supervised ML, but semantic embeddings can generalise across sentences with similar meanings even when their wording differs.

---

## 3. Qwen3-4B Zero-Shot

The small language model baseline uses:

```text
Qwen3-4B-Instruct-2507
Q4_K_M quantisation
```

through Ollama.

It receives:

```text
customer message
+
V2 class definitions
```

and selects exactly one V2 class.

It receives:

```text
0 new V2 labelled examples
```

and is not fine-tuned for this experiment.

---

## 4. Jev Zero-Shot

Jev receives exactly the same:

```text
customer message
+
frozen V2 class definitions
```

and also receives:

```text
0 new V2 labelled examples
```

The experiment uses the TypeSafe `jev-latest` endpoint, which resolved during the experiment to:

```text
jev-1.13.0
```

Jev returns:

- a typed class choice;
- probabilities across the available classes;
- a separate confidence value.

---

# Information Boundary

A key experiment requirement is preventing leakage of BANKING77's original intent labels.

The frozen model-facing test set contains only:

```text
id
text
```

The evaluator-side ground truth contains:

```text
id
text
original_intent
v1_label
v2_label
```

Models must read:

```text
data/prepared/test_600_inputs.csv
```

and must **not** read:

```text
data/prepared/test_600_ground_truth.csv
```

during inference.

---

# V2 Adaptation Protocol

For supervised ML, reusable historical data is retained for the clean V1 classes:

```text
TRANSFERS
TOP_UPS
CURRENCY
```

This gives:

```text
3,930 reusable historical examples
```

For the six V2 classes affected by taxonomy changes, two new-label budgets are tested:

```text
20 labels / affected class
100 labels / affected class
```

Equivalent totals:

```text
20/class  → 120 new labels
100/class → 600 new labels
```

Five independent random draws are generated for each budget.

The 20-example set is nested inside the corresponding 100-example set.

Because the reusable historical classes contain hundreds or thousands of examples while newly labelled classes contain only 20 or 100 examples, V2 Logistic Regression uses:

```python
class_weight="balanced"
```

---

# Evaluation Metrics

## Macro-F1

The primary quality metric.

F1 combines:

- **precision**: when a model predicts a class, how often is it correct?
- **recall**: of all real examples belonging to that class, how many does the model find?

Macro-F1 calculates F1 separately for every class and gives each class equal weight.

This is useful because the experiment classes are not equally sized.

Higher is better.

---

## Accuracy

The proportion of all test examples classified correctly.

Higher is better.

---

## p50 latency

The median end-to-end inference latency.

A p50 latency of 10 ms means:

- half the requests completed in 10 ms or less;
- half took longer.

Lower is better.

p95 latency is also recorded for tail-latency analysis.

---

# Repository Structure

```text
jev-experiment/
│
├── config/
│   ├── taxonomy.yaml
│   └── v2_classes.yaml
│
├── data/
│   ├── raw/
│   │   └── banking77/
│   │
│   ├── prepared/
│   │   ├── train_ground_truth.csv
│   │   ├── test_ground_truth.csv
│   │   ├── test_600_ground_truth.csv
│   │   ├── test_600_inputs.csv
│   │   ├── clean_v1_reusable.csv
│   │   │
│   │   └── adaptation/
│   │       ├── draw_01/
│   │       │   ├── v2_20.csv
│   │       │   └── v2_100.csv
│   │       └── ...
│   │
│   └── cache/
│
├── src/
│   ├── prepare_data.py
│   ├── validate_taxonomy.py
│   ├── build_experiment_data.py
│   ├── build_adaptation_data.py
│   ├── evaluate.py
│   ├── aggregate_results.py
│   ├── analyze_jev_calibration.py
│   │
│   └── models/
│       ├── tfidf_lr.py
│       ├── embedding_lr.py
│       ├── v2_ml.py
│       ├── slm_zero_shot.py
│       └── jev_zero_shot.py
│
├── results/
│   ├── predictions/
│   ├── raw/
│   ├── timings/
│   ├── metrics/
│   ├── summary/
│   ├── calibration/
│   └── figures/
│
├── blog/
│   └── notes.md
│
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

# Environment Setup

Python 3.11 or 3.12 is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn datasets sentence-transformers transformers torch matplotlib pyyaml python-dotenv requests
```

---

# Environment Variables

Copy:

```text
.env.example
```

to:

```text
.env
```

For Jev:

```text
TYPESAFE_API_KEY=
TYPESAFE_MODEL=
```

`TYPESAFE_MODEL` may be left empty. The experiment script discovers available models from the TypeSafe API and prefers:

```text
jev-latest
```

if available.

Do not commit `.env`.

---

# Reproducing the Experiment

## 1. Download BANKING77

```bash
python -m src.prepare_data
```

This downloads the pinned BANKING77 snapshot and stores the Hugging Face DatasetDict under:

```text
data/raw/banking77/
```

---

## 2. Validate the V1/V2 Taxonomy

```bash
python -m src.validate_taxonomy
```

Expected result:

```text
77 intents mapped to
7 V1 queues
9 V2 queues
```

---

## 3. Build Experiment Data

```bash
python -m src.build_experiment_data
```

This creates:

```text
data/prepared/
├── train_ground_truth.csv
├── test_ground_truth.csv
├── test_600_ground_truth.csv
└── test_600_inputs.csv
```

---

## 4. Build V2 Adaptation Sets

```bash
python -m src.build_adaptation_data
```

This creates:

- reusable clean historical data;
- five 20-label adaptation draws;
- five 100-label adaptation draws.

---

# V1 Baselines

## TF-IDF + Logistic Regression

Run:

```bash
python -m src.models.tfidf_lr
```

Evaluate:

```bash
python -m src.evaluate \
  --predictions results/predictions/tfidf_lr_v1.csv \
  --label-column v1_label \
  --name tfidf_lr_v1
```

Windows PowerShell:

```powershell
python -m src.evaluate --predictions results/predictions/tfidf_lr_v1.csv --label-column v1_label --name tfidf_lr_v1
```

---

## Embedding + Logistic Regression

Run:

```bash
python -m src.models.embedding_lr
```

Evaluate:

```powershell
python -m src.evaluate --predictions results/predictions/embedding_lr_v1.csv --label-column v1_label --name embedding_lr_v1
```

---

# V2 Supervised Adaptation

The shared V2 runner supports both models.

Example:

```powershell
python -m src.models.v2_ml --model tfidf --budget 20 --draw 1
```

or:

```powershell
python -m src.models.v2_ml --model embedding --budget 100 --draw 3
```

Parameters:

```text
--model   tfidf | embedding
--budget  20 | 100
--draw    1..5
```

Evaluate each generated prediction file using:

```powershell
python -m src.evaluate --predictions <prediction-file> --label-column v2_label --name <experiment-name>
```

---

## Aggregate V2 Results

After completing all TF-IDF and embedding V2 runs:

```bash
python -m src.aggregate_results
```

This produces:

```text
results/summary/
├── v2_ml_per_draw.csv
├── v2_ml_summary.csv
├── v2_ml_label_efficiency.csv
└── v2_ml_summary.json
```

---

# Qwen3-4B Zero-Shot Baseline

Install Ollama and pull:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Smoke test:

```bash
python -m src.models.slm_zero_shot --limit 10 --restart
```

Full run:

```bash
python -m src.models.slm_zero_shot --restart
```

If interrupted, resume with:

```bash
python -m src.models.slm_zero_shot
```

Evaluate:

```powershell
python -m src.evaluate --predictions results/predictions/slm_qwen3_4b_v2.csv --label-column v2_label --name slm_qwen3_4b_v2
```

---

# Jev Zero-Shot

Check models available to the TypeSafe account:

```bash
python -m src.models.jev_zero_shot --list-models
```

Smoke test:

```bash
python -m src.models.jev_zero_shot --limit 10 --restart
```

Full benchmark:

```bash
python -m src.models.jev_zero_shot --restart
```

If interrupted:

```bash
python -m src.models.jev_zero_shot
```

Evaluate:

```powershell
python -m src.evaluate --predictions results/predictions/jev_zero_shot_v2.csv --label-column v2_label --name jev_zero_shot_v2
```

---

# Jev Calibration Analysis

No additional API calls are required.

The Jev runner saves:

- selected class;
- class probabilities;
- API confidence;
- token usage;
- latency;
- raw response.

Run:

```bash
python -m src.analyze_jev_calibration
```

Outputs include:

```text
results/calibration/
├── jev_reliability_bins.csv
├── jev_confidence_quantiles.csv
├── jev_confidence_review.csv
├── jev_calibration_cases.csv
└── jev_calibration_summary.json
```

and:

```text
results/figures/
├── jev_reliability_diagram.png
├── jev_confidence_accuracy.png
└── jev_confidence_review.png
```

The calibration analysis separates:

1. **class probability calibration**;
2. **Jev API confidence as a risk-ranking signal**.

It also simulates sending the lowest-confidence 10%, 20%, and 30% of requests to human review.

---

# Main Results

## Stable V1 Taxonomy

| Model       |   Macro-F1 |   Accuracy |  p50 latency |
| ----------- | ---------: | ---------: | -----------: |
| TF-IDF + LR | **0.9588** | **0.9600** | **0.370 ms** |
| MiniLM + LR |     0.9508 |     0.9567 |    13.338 ms |

For the stable taxonomy, the simplest classical model was both more accurate and substantially faster.

---

# V2 Taxonomy Change

| Method             | New V2 labels |   Macro-F1 |   Accuracy |
| ------------------ | ------------: | ---------: | ---------: |
| TF-IDF + LR        |           120 |     0.6810 |     0.6843 |
| Qwen3-4B zero-shot |         **0** |     0.7749 |     0.7767 |
| Embedding + LR     |           120 |     0.8460 |     0.8450 |
| **Jev zero-shot**  |         **0** | **0.8954** | **0.8900** |
| TF-IDF + LR        |           600 |     0.8864 |     0.8837 |
| Embedding + LR     |           600 | **0.9065** | **0.9040** |

The experiment suggests that semantic embeddings are substantially more label-efficient than lexical TF-IDF under taxonomy change.

Jev reached a similar quality regime to supervised classifiers trained with hundreds of new labels while requiring no newly labelled V2 examples.

---

# Observed Latency

| Method         |  p50 latency |
| -------------- | -----------: |
| TF-IDF + LR    | **0.370 ms** |
| MiniLM + LR    |    13.338 ms |
| Jev API        |    684.89 ms |
| Qwen3-4B local |   1214.45 ms |

These numbers represent the actual deployment configurations used in the experiment.

They are **not hardware-normalised**.

In particular:

- Qwen ran locally;
- Jev used a hosted API.

---

# Jev API Cost

For the complete 600-request run:

```text
Input tokens: 325,972
API cost:     $0.013691
```

Equivalent approximate cost:

```text
$0.0228 per 1,000 classifications
```

This cost reflects the pricing and request configuration used during the experiment.

---

# Jev Probability and Confidence Findings

Jev classification accuracy:

```text
89.00%
```

Mean selected-class probability:

```text
94.43%
```

Expected Calibration Error:

```text
5.85%
```

This suggests that Jev was somewhat overconfident on this benchmark.

However, the separate API confidence value was useful as a risk-ranking signal.

If the lowest-confidence cases were routed to human review:

| Review workload       | Errors captured |
| --------------------- | --------------: |
| Lowest-confidence 10% |           36.4% |
| Lowest-confidence 20% |       **59.1%** |
| Lowest-confidence 30% |       **72.7%** |

This suggests a possible enterprise control pattern:

```text
Customer request
       ↓
      Jev
       ↓
decision + confidence
       │
       ├── high confidence → automated path
       │
       └── low confidence  → human review
```

---

# Key Engineering Takeaways

## Stable taxonomy

Use conventional supervised ML when:

- the classification taxonomy changes infrequently;
- labelled data is already available;
- very low latency matters.

A simple classifier may be difficult to justify replacing.

---

## Taxonomy change with limited labels

Semantic embeddings can provide a substantial advantage when only a small number of newly labelled examples are available.

---

## Taxonomy change with no immediate labels

A zero-shot decision model can reduce **adaptation latency**.

Instead of:

```text
collect labels
    ↓
train
    ↓
validate
    ↓
deploy
```

the system may operate directly from:

```text
new business definitions
        ↓
decision
```

This is the operating regime where Jev was most interesting in this experiment.

---

## Inference latency vs adaptation latency

The experiment highlights two different engineering concerns.

### Inference latency

How quickly can the system classify one request?

Traditional ML wins easily.

### Adaptation latency

How quickly can the system support a newly changed business decision?

Zero-shot approaches may have an advantage.

An enterprise architecture decision should consider both.

---

# Alternative Architecture: Predict Stable Fine-Grained Intents

The experiment assumes historical classifiers stored only the V1 operational business queue.

A different architecture could retain a more stable semantic layer:

```text
Customer request
       ↓
Fine-grained intent classifier
       ↓
Stable intent
       ↓
Business mapping
       ↓
Current operational queue
```

In this design, a business taxonomy change may require changing only the mapping rather than retraining the classifier.

This leads to an important conclusion:

> Sometimes the best solution to taxonomy change is not a more flexible model, but better information architecture.

---

# Limitations

This repository represents a practical engineering experiment rather than a comprehensive benchmark.

Important limitations include:

- BANKING77 contains short and relatively clean customer requests.
- Real enterprise tickets may be longer, noisy, multilingual, multi-intent, or contain irrelevant context.
- BANKING77 is public, so pretrained models may have encountered some of its data.
- The custom V1/V2 taxonomy reduces but does not eliminate possible dataset familiarity.
- The frozen evaluation set contains 600 examples.
- Small differences of a few percentage points should not be overinterpreted.
- Only one practical SLM configuration was tested.
- Qwen and Jev were deployed on different infrastructure.
- The V1 → V2 taxonomy migration is constructed rather than taken from an observed production migration.
- Jev probability calibration is specific to this dataset and should not be assumed to generalise.
- Human-review results are simulated.
- Actual human-review accuracy, latency, and cost were not measured.
- The experiment does not evaluate adversarial input, distribution shift, noisy text, or multilingual requests.

---

# Overall Conclusion

The experiment does **not** suggest that specialised decision models should replace traditional ML.

Instead, the approaches solve different operational problems.

For stable classification tasks with sufficient labelled data, traditional supervised ML remains fast, inexpensive, and highly effective.

When labelled data becomes scarce after a taxonomy change, semantic embeddings provide substantially better label efficiency than lexical TF-IDF.

A small language model provides immediate zero-shot adaptability, but that flexibility does not guarantee a precise business decision boundary.

Jev was most interesting in the period between:

```text
"The business changed the decision"
```

and:

```text
"We have enough newly labelled data to retrain a classifier"
```

In this experiment it achieved:

```text
Macro-F1:      0.8954
New V2 labels: 0
```

which was close to the embedding classifier trained with 600 new labels.

The architecture question is therefore not simply:

> **Which classifier is best?**

A more useful set of questions is:

> **How stable is the decision? How often will the business taxonomy change? How quickly can labelled data be collected? What latency is acceptable? And what should the software do when the model is uncertain?**

Those questions are likely to matter more in enterprise software than the model name itself.

---

# Related Blog Post

A longer discussion of the experiment and its implications is available here:

```text
https://pengbin.org/posts/jev-vs-ml-when-routing-rules-change/
```

---

# License

The experiment code can be licensed according to the repository's selected open-source license.

BANKING77 remains subject to its original dataset license and terms.

---