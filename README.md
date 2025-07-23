

# Extreme Multi-Label Classification with DiSMEC and DiSMEC++

## Introduction

Extreme Multi-label Classification (XMC) involves supervised learning tasks with an extremely large label space, often reaching hundreds of thousands or even millions of labels. This task is highly relevant in domains such as product categorization, web-scale recommendation systems, ad-tagging, and large-scale text classification.

This repository contains the implementation, experiment results, and usage documentation for applying DiSMEC and DiSMEC++ algorithms to standard datasets such as Eurlex-4K, AmazonCat-13K, and Amazon-670K, as part of my internship project at IIT Jodhpur

## About

This project aims to explore and benchmark DiSMEC and its improved variant DiSMEC++ for XMC problems.
Key highlights include:

* Evaluation on real-world large-scale datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html)
* Comparative analysis using P\@k, nDCG\@k, PSP\@k, PSnDCG\@k metrics
* Visualization of performance and memory trade-offs
* Model optimization for training time and storage size


## Implementation Details

### Algorithms Used

* DiSMEC – Distributed One-vs-All training with weight culling for sparse model generation.
* DiSMEC++ – An optimized version supporting multi-core CPU systems, memory-aware training, and dense features.

## How to Run

### Dataset Preparation

Download datasets from [XMC Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) and preprocess using provided Java scripts:

```bash
javac FeatureRemapper.java
java FeatureRemapper train.txt train-remapped.txt test.txt test-remapped.txt

javac TfIdfCalculator.java
java TfIdfCalculator train-remapped.txt train-tfidf.txt test-remapped.txt test-tfidf.txt

javac LabelRelabeler.java
java LabelRelabeler train-tfidf.txt train-final.txt test-tfidf.txt test-final.txt label-mapping.txt
```
### Training with DiSMEC++

```bash
./train train_eurlex.txt eurlex.model --augment-for-bias --normalize-instances --save-sparse-txt --weight-culling=0.01
```

This command trains the model and saves the sparse representation.

### Testing

```bash
./predict test_eurlex.txt eurlex.model predictions.txt --augment-for-bias --normalize-instances --topk=5 --save-metrics=metrics.json
```

Generates predictions and evaluation metrics in JSON format.

### Build DiSMEC++

```bash
mkdir build
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build --parallel
```

Make sure dependencies in the `deps/` directory are initialized (use `--recursive` when cloning).

---

## Results & Performance

| Dataset       | Method   | P\@1 | P\@3 | P\@5 | nDCG\@5 | Model Size | Train Time |
| ------------- | -------- | ---- | ---- | ---- | ------- | ---------- | ---------- |
| Eurlex-4K     | DiSMEC   | 82.4 | 68.5 | 57.7 | 66.7    | 0.23 GB    | 0.40 hr    |
|               | DiSMEC++ | 89.4 | 78.5 | 67.1 | 66.7    | 0.16 GB    | 0.20 hr    |
| AmazonCat-13K | DiSMEC   | 93.4 | 79.1 | 64.1 | 85.8    | 0.80 GB    | 4.70 hr    |
|               | DiSMEC++ | 94.9 | 82.1 | 62.1 | 82.1    | 0.54 GB    | 4.20 hr    |
| Amazon-670K   | DiSMEC   | 37.0 | 29.0 | 22.1 | 39.8    | -          | -          |
|               | DiSMEC++ | 45.7 | 36.7 | 38.1 | 49.5    | -          | -          |

## Directory Structure

```
.
├── build/                 # Build artifacts
├── dismec/               # DiSMEC C++ source code
├── preprocessing/        # Java preprocessing & evaluation scripts
├── test/                 # Evaluation scripts
├── datasets/             # Raw and processed data
└── report/               # Internship report and results
```

## References

1. Agrawal et al., WWW, 2013
2. Babbar & Schölkopf, WSDM, 2017
3. Schultheis & Babbar, arXiv:2109.13122
4. Bhatia et al., ACM WSDM, 2016
5. Weston et al., IJCAI, 2011
6. Lin et al., ICML, 2014
7. Mineiro & Karampatziakis, Preprint, 2015
8. Karampatziakis & Mineiro, Preprint, 2015
9. Balasubramanian & Lebanon, Preprint, 2012
10. Cisse et al., NIPS, 2013

