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

### Enterprise Attributes

* **Registered Capital:** measured in units of 10,000 RMB.
* **Firm Age:** calculated in months using June 1, 2025 as the reference date.
* **Industry Category:** original industry labels are mapped into major industry groups to reduce sparsity and improve generalization.

### Court Encoding

Court hierarchy information is encoded as:

| Court Level        | Encoding |
| ------------------ | -------: |
| Primary Court      |        0 |
| Intermediate Court |        1 |
| High Court         |        2 |
| Supreme Court      |        3 |

### Bankruptcy Labels

A binary label is used to represent enterprise status:

| Label | Meaning                         |
| ----- | ------------------------------- |
| 0     | Normal enterprise               |
| 1     | Bankrupt / High-risk enterprise |

### Data Sources

| File              | Description                         |
| ----------------- | ----------------------------------- |
| `basic_info.csv`  | Enterprise registration information |
| `cause.csv`       | Judicial case records               |
| `shareholder.csv` | Shareholder information             |
