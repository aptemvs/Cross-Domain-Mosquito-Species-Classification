# Technical Report Reproducibility Summary

## Configuration

- sample_rate: 8000
- n_fft: 512
- hop_length: 80
- win_length: 512
- n_mels: 64
- batch_size: 64
- eval_batch_size: 8
- train_crop_seconds: 2.0
- learning_rate: 0.001
- frequency_resolution_hz: 15.625000
- model_name: MTRCNNClassifier
- total_parameters: 221267
- trainable_parameters: 221267

## Dataset Statistics

| Split | Num clips | Mean frames | Median frames | Mode frames | Min frames | Max frames | Mean sec | Max sec | Iterations at batch 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 213647 | 66.891 | 63.0 | 63 | 4 | 29390 | 0.669 | 293.9 | 3339 |
| validation | 30516 | 65.083 | 63.0 | 63 | 9 | 6461 | 0.651 | 64.61 | 477 |
| test | 27217 | 204.347 | 63.0 | 63 | 8 | 15239 | 2.043 | 152.39 | 426 |

## Species Distribution Per Split

| Species | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Aedes aegypti | 64251 | 9177 | 8159 |
| Aedes albopictus | 14582 | 2083 | 1852 |
| Anopheles arabiensis | 16630 | 2375 | 2112 |
| Anopheles dirus | 76 | 11 | 40 |
| Anopheles gambiae | 37011 | 5287 | 4700 |
| Anopheles minimus | 395 | 56 | 99 |
| Anopheles stephensi | 525 | 75 | 74 |
| Culex pipiens | 23432 | 3347 | 2975 |
| Culex quinquefasciatus | 56745 | 8105 | 7206 |

## Domain Distribution Per Split

| Domain | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| D1 | 634 | 96 | 3335 |
| D2 | 230 | 30 | 524 |
| D3 | 364 | 53 | 262 |
| D4 | 80 | 3 | 117 |
| D5 | 212339 | 30334 | 22979 |

## Official Seen/Unseen Test Counts

| Species | Seen test clips | Unseen test clips |
| --- | ---: | ---: |
| Ae.aeg | 7967 | 192 |
| Ae.alb | 1433 | 419 |
| Cx.qui | 6534 | 672 |
| An.gam | 3882 | 818 |
| An.ara | 292 | 1820 |
| An.dir | 0 | 40 |
| Cx.pip | 2907 | 68 |
| An.min | 0 | 99 |
| An.ste | 0 | 74 |

## Multi-Seed Summary

| Metric | Validation mean | Validation std | Test mean | Test std |
| --- | ---: | ---: | ---: | ---: |
| species_accuracy | 0.90561 | 0.007648 | 0.782695 | 0.006792 |
| species_balanced_accuracy | 0.852203 | 0.010827 | 0.541118 | 0.014039 |
| domain_accuracy | 0.999151 | 9e-05 | 0.922328 | 0.023025 |
| loss | 0.258336 | 0.019666 | 1.636667 | 0.142887 |
| epochs_ran | 23.6 | 9.850888 | 23.6 | 9.850888 |
| BA_seen | n/a | n/a | 0.880582 | 0.010791 |
| BA_unseen | n/a | n/a | 0.175113 | 0.019705 |
| DSG | n/a | n/a | 0.705469 | 0.024794 |

## Per-Domain Test Species Balanced Accuracy

| Domain | Mean | Std |
| --- | ---: | ---: |
| D1 | 0.428582 | 0.025449 |
| D2 | 0.339512 | 0.079249 |
| D3 | 0.316748 | 0.075042 |
| D4 | 0.117568 | 0.056829 |
| D5 | 0.880932 | 0.010795 |

## Per-Species Official Results (Best Model)

| Species | BA_seen mean | BA_seen std | BA_unseen mean | BA_unseen std | DSG mean | DSG std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ae.aeg | 0.91205 | 0.012003 | 0.045833 | 0.094866 | 0.866216 | 0.100104 |
| Ae.alb | 0.86127 | 0.022291 | 0.063484 | 0.048092 | 0.797786 | 0.065922 |
| Cx.qui | 0.956413 | 0.014612 | 0.001042 | 0.002666 | 0.955371 | 0.014469 |
| An.gam | 0.898248 | 0.024939 | 0.000122 | 0.000367 | 0.898126 | 0.025056 |
| An.ara | 0.725342 | 0.039932 | 0.002473 | 0.004817 | 0.72287 | 0.039188 |
| An.dir | None | None | 0.0 | 0.0 | None | None |
| Cx.pip | 0.930169 | 0.017046 | 0.804412 | 0.034832 | 0.125757 | 0.036743 |
| An.min | None | None | 0.238384 | 0.069278 | None | None |
| An.ste | None | None | 0.42027 | 0.140884 | None | None |

## Per-Species Official Results (Final Model)

| Species | BA_seen mean | BA_seen std | BA_unseen mean | BA_unseen std | DSG mean | DSG std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ae.aeg | 0.917096 | 0.010925 | 0.027083 | 0.052694 | 0.890012 | 0.056065 |
| Ae.alb | 0.851291 | 0.014785 | 0.073986 | 0.109583 | 0.777305 | 0.114493 |
| Cx.qui | 0.960836 | 0.007923 | 0.001488 | 0.002399 | 0.959348 | 0.00767 |
| An.gam | 0.919114 | 0.015659 | 0.000122 | 0.000367 | 0.918992 | 0.015855 |
| An.ara | 0.718836 | 0.043547 | 0.007418 | 0.014953 | 0.711418 | 0.054586 |
| An.dir | None | None | 0.005 | 0.015 | None | None |
| Cx.pip | 0.9258 | 0.02422 | 0.811765 | 0.028516 | 0.114035 | 0.038237 |
| An.min | None | None | 0.225253 | 0.061117 | None | None |
| An.ste | None | None | 0.381081 | 0.150698 | None | None |

## Seed-Level Training Summary

| Seed | Epochs ran | Best epoch | Best val species BA | Final val species BA | Test species BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 19 | 14 | 0.851930 | 0.848161 | 0.536415 |
| 1024 | 19 | 14 | 0.850772 | 0.843538 | 0.551956 |
| 1234 | 38 | 33 | 0.868112 | 0.861915 | 0.526124 |
| 2023 | 44 | 39 | 0.861484 | 0.853112 | 0.548127 |
| 2024 | 16 | 11 | 0.850071 | 0.835891 | 0.541383 |
| 2048 | 30 | 25 | 0.864304 | 0.844293 | 0.551909 |
| 3407 | 14 | 8 | 0.834317 | 0.814981 | 0.506582 |
| 4096 | 23 | 18 | 0.833371 | 0.828798 | 0.547047 |
| 8192 | 14 | 7 | 0.853628 | 0.801371 | 0.547305 |
| 10086 | 19 | 14 | 0.854042 | 0.850749 | 0.554332 |

## Seed-Level Notes

- best test species balanced accuracy: seed 10086 -> 0.554332
- worst test species balanced accuracy: seed 3407 -> 0.506582
- average epochs actually run: 23.60
- average gap between best validation species BA and final validation species BA: 0.013922

## Recommended Report Items

- dataset split sizes and feature-length statistics
- species and domain distribution per split
- feature extraction settings and normalization procedure
- frequency resolution and mel frontend definition
- model architecture, parameter count, and dual-head training objective
- training setup, early stopping rule, optimizer, batch sizes, and device policy
- train/validation loss curves and validation metric curves
- validation and test metrics over all seeds
- seed-level epoch counts, best epoch, and checkpoint selection rule
- per-domain species balanced accuracy tables
- saved prediction files for additional reproducible metrics
