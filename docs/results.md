# Results (auto-generated)

Modelling table `v81f4e629`; cleaning counts: `{'input': 4856989, 'dedupe_case_id': 4856989, 'drop_null_keys': 4856989, 'drop_filed_after_extraction': 4856989, 'drop_decision_before_filing': 4823868, 'drop_decision_after_extraction': 4823432, 'drop_pandemic_affected': 4823189, 'final': 4823189, 'events': 4167876, 'censored': 655313}`

## Survival models by validation scheme
| split | model | C-index | IBS | AUC@24m | Brier@24m | TD-AUC@12 | TD-AUC@36 | TD-AUC@60 | coverage | width |
|---|---|---|---|---|---|---|---|---|---|---|
| random | cox | 0.707 | 0.170 | 0.766 | 0.194 | 0.774 | 0.773 | 0.788 | 0.900 | 0.929 |
| random | rsf | 0.691 | 0.178 | 0.753 | 0.204 | 0.761 | 0.759 | 0.772 | 0.902 | 0.961 |
| temporal | cox | 0.691 | 0.222 | 0.744 | 0.205 | 0.746 | 0.701 | nan | 0.866 | 0.901 |
| temporal | rsf | 0.686 | 0.226 | 0.745 | 0.207 | 0.739 | 0.708 | nan | 0.875 | 0.947 |
| spatial | cox | 0.702 | 0.175 | 0.766 | 0.194 | 0.783 | 0.758 | 0.750 | 0.912 | 0.936 |
| spatial | rsf | 0.696 | 0.183 | 0.759 | 0.205 | 0.781 | 0.752 | 0.749 | 0.890 | 0.967 |

## Baseline replication vs published
| metric | published | replicated | diff |
|---|---|---|---|
| auc_24m | 0.890 | 0.794 | -0.096 |
| ontime_rmse | 4.900 | 6.157 | 1.257 |
| ontime_mae | 3.330 | 4.925 | 1.595 |
| delayed_rmse | 13.790 | 16.430 | 2.640 |
| delayed_mae | 10.740 | 13.276 | 2.536 |

## Landmark models (temporal split)
| landmark | model | n_cohort | n_known_label | auc_24m | brier_24m |
|---|---|---|---|---|---|
| filing | at-filing model | 22918 | 22918 | 0.744 | 0.205 |
| filing | landmark model | 22918 | 22918 | 0.751 | 0.204 |
| first_hearing | at-filing model | 21329 | 21329 | 0.750 | 0.203 |
| first_hearing | landmark model | 21329 | 21329 | 0.754 | 0.202 |
| 6m | at-filing model | 15821 | 15821 | 0.658 | 0.229 |
| 6m | landmark model | 15821 | 15821 | 0.677 | 0.227 |
| 12m | at-filing model | 13186 | 13186 | 0.642 | 0.195 |
| 12m | landmark model | 13186 | 13186 | 0.650 | 0.192 |

## Permutation importance (feature groups)
| feature_group | c_index_drop |
|---|---|
| case_type | 0.107 |
| act | 0.067 |
| court | 0.032 |
| congestion | 0.005 |
| season | 0.000 |

## Ablation (judge features unavailable -> documented exclusion)
| removed_group | c_index |
|---|---|
| none (full model) | 0.688 |
| act | 0.660 |
| case_type | 0.632 |
| congestion | 0.690 |
| court | 0.683 |
| season | 0.688 |
| judge features | nan |

## C-index by court size
| court_size | n | c_index |
|---|---|---|
| small | 7639 | 0.674 |
| medium | 7639 | 0.696 |
| large | 7640 | 0.693 |
