# TAGDA



## Complexity

| Complexity Type        | TAGDA                  |
| ---------------------- | ---------------------- |
| Time Complexity        | (O(LEd + Nd^2 + M^2d)) |
| Space Complexity       | (O(Nd + E + M^2))      |
| Labeled Samples        | (O(N_s))               |
| Unlabeled Samples      | (O(N_t))               |

where (N) is the number of nodes, (E) is the number of edges, (L) is the number of GNN layers, (d) is the hidden dimension, and (M) is the number of sampled nodes used for contrastive learning.



## Dataset Split

To ensure a fair evaluation, we adopt a stratified random split strategy at the enterprise level.

* **Split ratio:** 60% training, 20% validation, and 20% testing.
* **Random seed:** 42.
* **Split unit:** enterprise ID (`com_id`).
* **Stratification:** the bankruptcy label distribution is preserved across training, validation, and test sets.
* **No overlap:** enterprises appearing in one split do not appear in any other split.
* **Multi-table consistency:** all associated records in `basic_info.csv`, `cause.csv`, and `shareholder.csv` are assigned to the same split according to their enterprise IDs.

This strategy avoids information leakage between training and evaluation data while maintaining consistent class distributions across all subsets.


## Dataset Collection

The enterprise data used in this project were collected from publicly accessible enterprise information platforms in Sichuan Province, China. The dataset covers micro and small enterprises (MSEs) from two regions:

| Region                                      | Enterprises |
| ------------------------------------------- | ----------: |
| Chengdu                                     |      24,998 |
| Aba Tibetan and Qiang Autonomous Prefecture |      43,835 |
| Total                                       |      68,833 |

Each enterprise record contains:

* Basic registration information
* Judicial case records
* Shareholder information
* Bankruptcy labels

The Chengdu dataset is used as the primary evaluation dataset, while the Aba dataset is used to evaluate the cross-region generalization capability of TAGDA.

## Data Preprocessing

Several preprocessing steps were performed before graph construction:

###### Enterprise Attributes

* **Registered Capital:** measured in units of 10,000 RMB.
* **Firm Age:** calculated in months using June 1, 2025 as the reference date.
* **Industry Category:** original industry labels are mapped into major industry groups to reduce sparsity and improve generalization.

##### Court Encoding

Court hierarchy information is encoded as:

| Court Level        | Encoding |
| ------------------ | -------: |
| Primary Court      |        0 |
| Intermediate Court |        1 |
| High Court         |        2 |
| Supreme Court      |        3 |

##### Bankruptcy Labels

A binary label is used to represent enterprise status:

| Label | Meaning                         |
| ----- | ------------------------------- |
| 0     | Normal enterprise               |
| 1     | Bankrupt / High-risk enterprise |

##### Data Sources

| File              | Description                         |
| ----------------- | ----------------------------------- |
| `basic_info.csv`  | Enterprise registration information |
| `cause.csv`       | Judicial case records               |
| `shareholder.csv` | Shareholder information             |

## Environment

Experiments were conducted under the following environment:

| Component         | Version    |
| ----------------- | ---------- |
| OS                | Windows 10 |
| Python            | 3.12       |
| CUDA              | 12.9       |
| PyTorch           | 2.9.0      |
| PyTorch Geometric | 2.6.1      |
| NumPy             | 1.26.4     |
| Pandas            | 2.3.2      |
| Scikit-learn      | 1.7.1      |

##### Main Dependencies

* torch
* torch-geometric
* numpy
* pandas
* scikit-learn
* scipy
* xgboost
* networkx
* tqdm

* ## Evaluation Metrics

To comprehensively evaluate bankruptcy prediction performance under severe class imbalance, we report the following metrics:

| Metric            | Description                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| AUC               | Area Under the ROC Curve. Measures the ranking ability of the model across different classification thresholds. |
| AUPRC             | Area Under the Precision-Recall Curve. More informative than AUC for highly imbalanced datasets.                |
| F1pos             | F1-score of the positive class (high-risk enterprises).                                                         |
| Sensitivity (TPR) | True Positive Rate, measuring the ability to correctly identify high-risk enterprises.                          |
| Specificity (TNR) | True Negative Rate, measuring the ability to correctly identify normal enterprises.                             |

The metrics are computed based on the confusion matrix:

|                    | Actual Positive | Actual Negative |
| ------------------ | --------------- | --------------- |
| Predicted Positive | TP              | FP              |
| Predicted Negative | FN              | TN              |

where TP, TN, FP, and FN denote true positives, true negatives, false positives, and false negatives, respectively.


## Computational Cost

All experiments were conducted on the following hardware platform:

| Component | Specification              |
| --------- | -------------------------- |
| OS        | Windows 10 Professional    |
| CPU       | Intel Core i5-13400F       |
| RAM       | 32 GB                      |
| GPU       | NVIDIA GeForce RTX 5060 Ti |
| Storage   | 1 TB SSD                   |

##### Training Efficiency

Under the default configuration:

* Average training time per epoch: approximately **3 minutes**
* GPU memory: approximately **8–12 GB** (depending on graph size and batch configuration)
* Training is performed on a single GPU

The proposed TAGDA framework can be trained on a consumer-grade GPU without requiring distributed training or high-performance computing resources.


