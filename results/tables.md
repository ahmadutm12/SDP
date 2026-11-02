
### Table 6 MCC

| Dataset | Approach | DT | RF | XGBoost | SVM |
|---|---|---|---|---|---|
| CM1 | No correction | 0.069 ± 0.214 | 0.047 ± 0.171 | 0.127 ± 0.202 | 0.269 ± 0.140 |
| CM1 | Duplicate-bug-report | 0.075 ± 0.144 | 0.040 ± 0.160 | 0.110 ± 0.198 | 0.273 ± 0.152 |
| CM1 | GEO | 0.041 ± 0.162 | 0.044 ± 0.167 | 0.075 ± 0.187 | 0.200 ± 0.156 |
| CM1 | Information-theory | 0.127 ± 0.158 | 0.085 ± 0.164 | 0.097 ± 0.160 | 0.237 ± 0.161 |
| JM1 | No correction | 0.131 ± 0.034 | 0.245 ± 0.033 | 0.220 ± 0.033 | 0.253 ± 0.032 |
| JM1 | Duplicate-bug-report | 0.138 ± 0.037 | 0.251 ± 0.030 | 0.220 ± 0.030 | 0.258 ± 0.030 |
| JM1 | GEO | 0.130 ± 0.035 | 0.225 ± 0.035 | 0.192 ± 0.040 | 0.243 ± 0.031 |
| JM1 | Information-theory | 0.123 ± 0.037 | 0.226 ± 0.035 | 0.198 ± 0.037 | 0.255 ± 0.029 |
| KC1 | No correction | 0.149 ± 0.091 | 0.241 ± 0.079 | 0.254 ± 0.086 | 0.253 ± 0.091 |
| KC1 | Duplicate-bug-report | 0.150 ± 0.084 | 0.245 ± 0.086 | 0.267 ± 0.094 | 0.255 ± 0.088 |
| KC1 | GEO | 0.157 ± 0.098 | 0.223 ± 0.073 | 0.219 ± 0.073 | 0.227 ± 0.105 |
| KC1 | Information-theory | 0.126 ± 0.096 | 0.194 ± 0.090 | 0.178 ± 0.096 | 0.234 ± 0.099 |
| PC1 | No correction | 0.229 ± 0.150 | 0.351 ± 0.163 | 0.349 ± 0.170 | 0.304 ± 0.094 |
| PC1 | Duplicate-bug-report | 0.240 ± 0.136 | 0.324 ± 0.164 | 0.322 ± 0.162 | 0.282 ± 0.091 |
| PC1 | GEO | 0.264 ± 0.122 | 0.343 ± 0.137 | 0.315 ± 0.145 | 0.269 ± 0.089 |
| PC1 | Information-theory | 0.248 ± 0.123 | 0.311 ± 0.119 | 0.319 ± 0.161 | 0.298 ± 0.098 |

### Table 7 ROC-AUC

| Dataset | Approach | DT | RF | XGBoost | SVM |
|---|---|---|---|---|---|
| CM1 | No correction | 0.540 ± 0.129 | 0.704 ± 0.106 | 0.690 ± 0.139 | 0.753 ± 0.101 |
| CM1 | Duplicate-bug-report | 0.544 ± 0.085 | 0.713 ± 0.101 | 0.701 ± 0.126 | 0.756 ± 0.100 |
| CM1 | GEO | 0.524 ± 0.091 | 0.668 ± 0.121 | 0.639 ± 0.143 | 0.723 ± 0.098 |
| CM1 | Information-theory | 0.571 ± 0.092 | 0.697 ± 0.115 | 0.643 ± 0.128 | 0.721 ± 0.108 |
| JM1 | No correction | 0.568 ± 0.018 | 0.685 ± 0.020 | 0.664 ± 0.018 | 0.692 ± 0.020 |
| JM1 | Duplicate-bug-report | 0.572 ± 0.019 | 0.686 ± 0.020 | 0.664 ± 0.019 | 0.694 ± 0.019 |
| JM1 | GEO | 0.571 ± 0.021 | 0.671 ± 0.022 | 0.647 ± 0.023 | 0.686 ± 0.018 |
| JM1 | Information-theory | 0.563 ± 0.019 | 0.669 ± 0.023 | 0.649 ± 0.022 | 0.690 ± 0.019 |
| KC1 | No correction | 0.574 ± 0.048 | 0.674 ± 0.048 | 0.667 ± 0.054 | 0.674 ± 0.059 |
| KC1 | Duplicate-bug-report | 0.577 ± 0.043 | 0.676 ± 0.050 | 0.668 ± 0.054 | 0.673 ± 0.060 |
| KC1 | GEO | 0.572 ± 0.053 | 0.662 ± 0.049 | 0.652 ± 0.055 | 0.667 ± 0.057 |
| KC1 | Information-theory | 0.558 ± 0.056 | 0.647 ± 0.048 | 0.622 ± 0.061 | 0.671 ± 0.060 |
| PC1 | No correction | 0.645 ± 0.095 | 0.840 ± 0.066 | 0.835 ± 0.065 | 0.822 ± 0.062 |
| PC1 | Duplicate-bug-report | 0.644 ± 0.083 | 0.841 ± 0.065 | 0.832 ± 0.067 | 0.821 ± 0.062 |
| PC1 | GEO | 0.660 ± 0.077 | 0.826 ± 0.075 | 0.803 ± 0.081 | 0.788 ± 0.081 |
| PC1 | Information-theory | 0.655 ± 0.079 | 0.829 ± 0.068 | 0.815 ± 0.082 | 0.791 ± 0.084 |

### Table 8 Recall

| Dataset | Approach | DT | RF | XGBoost | SVM |
|---|---|---|---|---|---|
| CM1 | No correction | 0.232 ± 0.244 | 0.168 ± 0.184 | 0.222 ± 0.188 | 0.610 ± 0.181 |
| CM1 | Duplicate-bug-report | 0.241 ± 0.174 | 0.165 ± 0.175 | 0.218 ± 0.203 | 0.619 ± 0.199 |
| CM1 | GEO | 0.215 ± 0.201 | 0.177 ± 0.183 | 0.203 ± 0.185 | 0.548 ± 0.217 |
| CM1 | Information-theory | 0.299 ± 0.179 | 0.253 ± 0.190 | 0.279 ± 0.201 | 0.647 ± 0.203 |
| JM1 | No correction | 0.369 ± 0.032 | 0.348 ± 0.032 | 0.303 ± 0.032 | 0.528 ± 0.035 |
| JM1 | Duplicate-bug-report | 0.374 ± 0.032 | 0.351 ± 0.028 | 0.297 ± 0.027 | 0.516 ± 0.035 |
| JM1 | GEO | 0.375 ± 0.033 | 0.361 ± 0.035 | 0.316 ± 0.039 | 0.519 ± 0.048 |
| JM1 | Information-theory | 0.360 ± 0.032 | 0.356 ± 0.028 | 0.318 ± 0.035 | 0.504 ± 0.033 |
| KC1 | No correction | 0.407 ± 0.082 | 0.414 ± 0.084 | 0.404 ± 0.081 | 0.496 ± 0.096 |
| KC1 | Duplicate-bug-report | 0.405 ± 0.084 | 0.414 ± 0.087 | 0.415 ± 0.090 | 0.506 ± 0.091 |
| KC1 | GEO | 0.404 ± 0.088 | 0.397 ± 0.076 | 0.391 ± 0.084 | 0.494 ± 0.089 |
| KC1 | Information-theory | 0.395 ± 0.094 | 0.403 ± 0.081 | 0.385 ± 0.077 | 0.537 ± 0.089 |
| PC1 | No correction | 0.391 ± 0.191 | 0.414 ± 0.190 | 0.389 ± 0.185 | 0.711 ± 0.169 |
| PC1 | Duplicate-bug-report | 0.374 ± 0.170 | 0.391 ± 0.197 | 0.369 ± 0.192 | 0.669 ± 0.159 |
| PC1 | GEO | 0.414 ± 0.156 | 0.426 ± 0.154 | 0.363 ± 0.136 | 0.686 ± 0.158 |
| PC1 | Information-theory | 0.403 ± 0.165 | 0.414 ± 0.145 | 0.389 ± 0.161 | 0.731 ± 0.172 |

### Table 9 F1

| Dataset | Approach | DT | RF | XGBoost | SVM |
|---|---|---|---|---|---|
| CM1 | No correction | 0.182 ± 0.183 | 0.147 ± 0.153 | 0.213 ± 0.176 | 0.351 ± 0.108 |
| CM1 | Duplicate-bug-report | 0.187 ± 0.124 | 0.143 ± 0.145 | 0.199 ± 0.178 | 0.353 ± 0.113 |
| CM1 | GEO | 0.157 ± 0.137 | 0.150 ± 0.147 | 0.178 ± 0.163 | 0.297 ± 0.115 |
| CM1 | Information-theory | 0.233 ± 0.130 | 0.195 ± 0.133 | 0.208 ± 0.134 | 0.323 ± 0.116 |
| JM1 | No correction | 0.341 ± 0.026 | 0.391 ± 0.028 | 0.358 ± 0.030 | 0.443 ± 0.023 |
| JM1 | Duplicate-bug-report | 0.346 ± 0.027 | 0.395 ± 0.025 | 0.355 ± 0.026 | 0.445 ± 0.023 |
| JM1 | GEO | 0.342 ± 0.026 | 0.386 ± 0.027 | 0.351 ± 0.033 | 0.436 ± 0.023 |
| JM1 | Information-theory | 0.334 ± 0.027 | 0.384 ± 0.026 | 0.355 ± 0.029 | 0.441 ± 0.022 |
| KC1 | No correction | 0.381 ± 0.066 | 0.427 ± 0.064 | 0.430 ± 0.068 | 0.458 ± 0.068 |
| KC1 | Duplicate-bug-report | 0.381 ± 0.064 | 0.429 ± 0.069 | 0.441 ± 0.077 | 0.462 ± 0.064 |
| KC1 | GEO | 0.384 ± 0.071 | 0.413 ± 0.059 | 0.408 ± 0.062 | 0.445 ± 0.074 |
| KC1 | Information-theory | 0.366 ± 0.073 | 0.401 ± 0.067 | 0.388 ± 0.068 | 0.457 ± 0.069 |
| PC1 | No correction | 0.289 ± 0.132 | 0.391 ± 0.151 | 0.384 ± 0.161 | 0.326 ± 0.066 |
| PC1 | Duplicate-bug-report | 0.297 ± 0.124 | 0.365 ± 0.149 | 0.361 ± 0.155 | 0.311 ± 0.066 |
| PC1 | GEO | 0.319 ± 0.107 | 0.387 ± 0.120 | 0.358 ± 0.130 | 0.298 ± 0.063 |
| PC1 | Information-theory | 0.304 ± 0.109 | 0.360 ± 0.106 | 0.364 ± 0.145 | 0.317 ± 0.073 |

### Table S1a Precision

| Dataset | Approach | DT | RF | XGBoost | SVM |
|---|---|---|---|---|---|
| CM1 | No correction | 0.158 ± 0.161 | 0.144 ± 0.156 | 0.238 ± 0.227 | 0.251 ± 0.088 |
| CM1 | Duplicate-bug-report | 0.163 ± 0.116 | 0.135 ± 0.137 | 0.203 ± 0.193 | 0.251 ± 0.088 |
| CM1 | GEO | 0.138 ± 0.159 | 0.147 ± 0.154 | 0.172 ± 0.166 | 0.208 ± 0.085 |
| CM1 | Information-theory | 0.208 ± 0.140 | 0.181 ± 0.167 | 0.177 ± 0.124 | 0.219 ± 0.089 |
| JM1 | No correction | 0.318 ± 0.024 | 0.447 ± 0.031 | 0.439 ± 0.030 | 0.383 ± 0.023 |
| JM1 | Duplicate-bug-report | 0.322 ± 0.028 | 0.453 ± 0.030 | 0.444 ± 0.032 | 0.391 ± 0.019 |
| JM1 | GEO | 0.315 ± 0.026 | 0.417 ± 0.035 | 0.398 ± 0.039 | 0.377 ± 0.022 |
| JM1 | Information-theory | 0.312 ± 0.027 | 0.420 ± 0.034 | 0.405 ± 0.039 | 0.392 ± 0.020 |
| KC1 | No correction | 0.362 ± 0.064 | 0.450 ± 0.065 | 0.469 ± 0.077 | 0.430 ± 0.064 |
| KC1 | Duplicate-bug-report | 0.364 ± 0.061 | 0.453 ± 0.069 | 0.478 ± 0.077 | 0.430 ± 0.064 |
| KC1 | GEO | 0.371 ± 0.071 | 0.436 ± 0.058 | 0.435 ± 0.061 | 0.408 ± 0.075 |
| KC1 | Information-theory | 0.345 ± 0.064 | 0.406 ± 0.071 | 0.398 ± 0.081 | 0.401 ± 0.066 |
| PC1 | No correction | 0.235 ± 0.110 | 0.396 ± 0.165 | 0.421 ± 0.208 | 0.213 ± 0.045 |
| PC1 | Duplicate-bug-report | 0.255 ± 0.111 | 0.375 ± 0.180 | 0.383 ± 0.154 | 0.205 ± 0.045 |
| PC1 | GEO | 0.270 ± 0.099 | 0.379 ± 0.159 | 0.384 ± 0.192 | 0.192 ± 0.044 |
| PC1 | Information-theory | 0.253 ± 0.098 | 0.338 ± 0.127 | 0.369 ± 0.187 | 0.205 ± 0.054 |

### Table S1b Accuracy

| Dataset | Approach | DT | RF | XGBoost | SVM |
|---|---|---|---|---|---|
| CM1 | No correction | 0.783 ± 0.061 | 0.807 ± 0.056 | 0.825 ± 0.056 | 0.751 ± 0.063 |
| CM1 | Duplicate-bug-report | 0.781 ± 0.052 | 0.805 ± 0.053 | 0.823 ± 0.055 | 0.750 ± 0.064 |
| CM1 | GEO | 0.769 ± 0.062 | 0.795 ± 0.060 | 0.802 ± 0.072 | 0.719 ± 0.069 |
| CM1 | Information-theory | 0.786 ± 0.065 | 0.778 ± 0.062 | 0.777 ± 0.062 | 0.697 ± 0.080 |
| JM1 | No correction | 0.679 ± 0.015 | 0.756 ± 0.011 | 0.756 ± 0.009 | 0.702 ± 0.017 |
| JM1 | Duplicate-bug-report | 0.681 ± 0.018 | 0.758 ± 0.011 | 0.758 ± 0.011 | 0.710 ± 0.012 |
| JM1 | GEO | 0.675 ± 0.019 | 0.741 ± 0.018 | 0.738 ± 0.016 | 0.698 ± 0.019 |
| JM1 | Information-theory | 0.677 ± 0.018 | 0.743 ± 0.015 | 0.740 ± 0.018 | 0.712 ± 0.013 |
| KC1 | No correction | 0.658 ± 0.042 | 0.715 ± 0.033 | 0.724 ± 0.036 | 0.697 ± 0.038 |
| KC1 | Duplicate-bug-report | 0.660 ± 0.038 | 0.717 ± 0.034 | 0.729 ± 0.035 | 0.694 ± 0.043 |
| KC1 | GEO | 0.665 ± 0.044 | 0.709 ± 0.030 | 0.708 ± 0.028 | 0.679 ± 0.048 |
| KC1 | Information-theory | 0.647 ± 0.042 | 0.689 ± 0.040 | 0.684 ± 0.046 | 0.669 ± 0.044 |
| PC1 | No correction | 0.861 ± 0.031 | 0.909 ± 0.023 | 0.913 ± 0.025 | 0.784 ± 0.039 |
| PC1 | Duplicate-bug-report | 0.873 ± 0.027 | 0.904 ± 0.026 | 0.910 ± 0.023 | 0.782 ± 0.039 |
| PC1 | GEO | 0.870 ± 0.032 | 0.901 ± 0.025 | 0.904 ± 0.027 | 0.760 ± 0.045 |
| PC1 | Information-theory | 0.867 ± 0.029 | 0.891 ± 0.029 | 0.899 ± 0.031 | 0.766 ± 0.050 |

### Table 10 dMCC (A12) vs none; * Holm p<0.05

| Dataset | Comparison | DT | RF | XGBoost | SVM |
|---|---|---|---|---|---|
| CM1 | Duplicate-bug-report vs. no correction | +0.006 (0.53) | -0.006 (0.49) | -0.017 (0.48) | +0.003 (0.51) |
| CM1 | GEO vs. no correction | -0.028 (0.49) | -0.002 (0.49) | -0.052 (0.43) | -0.070 (0.38)* |
| CM1 | Information-theory vs. no correction | +0.058 (0.60) | +0.039 (0.57) | -0.030 (0.46) | -0.032 (0.45) |
| JM1 | Duplicate-bug-report vs. no correction | +0.007 (0.56) | +0.006 (0.54) | +0.001 (0.50) | +0.005 (0.55) |
| JM1 | GEO vs. no correction | -0.002 (0.48) | -0.020 (0.33)* | -0.027 (0.30)* | -0.010 (0.42)* |
| JM1 | Information-theory vs. no correction | -0.009 (0.43) | -0.019 (0.35)* | -0.021 (0.33)* | +0.002 (0.54) |
| KC1 | Duplicate-bug-report vs. no correction | +0.002 (0.48) | +0.003 (0.52) | +0.013 (0.54) | +0.003 (0.49) |
| KC1 | GEO vs. no correction | +0.009 (0.50) | -0.019 (0.43) | -0.035 (0.38)* | -0.026 (0.43)* |
| KC1 | Information-theory vs. no correction | -0.023 (0.40) | -0.047 (0.35)* | -0.076 (0.28)* | -0.019 (0.45)* |
| PC1 | Duplicate-bug-report vs. no correction | +0.011 (0.53) | -0.027 (0.45) | -0.027 (0.43) | -0.023 (0.43) |
| PC1 | GEO vs. no correction | +0.035 (0.57) | -0.008 (0.47) | -0.034 (0.41) | -0.035 (0.40) |
| PC1 | Information-theory vs. no correction | +0.019 (0.54) | -0.039 (0.41)* | -0.030 (0.42) | -0.006 (0.49) |

### Friedman (MCC)

| Dataset | Classifier | chi2 | p |
|---|---|---|---|
| CM1 | DT | 6.45 | 0.0918 |
| CM1 | RF | 0.80 | 0.848 |
| CM1 | SVM | 12.14 | 0.00691 |
| CM1 | XGBoost | 5.58 | 0.134 |
| JM1 | DT | 6.19 | 0.103 |
| JM1 | RF | 31.59 | 6.38e-07 |
| JM1 | SVM | 20.13 | 0.00016 |
| JM1 | XGBoost | 24.19 | 2.28e-05 |
| KC1 | DT | 5.38 | 0.146 |
| KC1 | RF | 22.82 | 4.4e-05 |
| KC1 | SVM | 16.65 | 0.000836 |
| KC1 | XGBoost | 28.69 | 2.6e-06 |
| PC1 | DT | 0.75 | 0.86 |
| PC1 | RF | 8.97 | 0.0297 |
| PC1 | SVM | 8.82 | 0.0318 |
| PC1 | XGBoost | 7.32 | 0.0624 |

### Selection summary

|                       |   n_selected_mean |   n_selected_min |   n_selected_max |   removed_mean |   removed_min |   removed_max |   sel_sec_mean |   geo_evals |
|:----------------------|------------------:|-----------------:|-----------------:|---------------:|--------------:|--------------:|---------------:|------------:|
| ('CM1', 'duplicate')  |             21    |               21 |               21 |           1.64 |             0 |             2 |          0.004 |      nan    |
| ('CM1', 'geo')        |             10.22 |                4 |               15 |           0    |             0 |             0 |          0.752 |       40.36 |
| ('CM1', 'infotheory') |              8.3  |                5 |               15 |           0    |             0 |             0 |          0.063 |      nan    |
| ('CM1', 'none')       |             21    |               21 |               21 |           0    |             0 |             0 |          0.002 |      nan    |
| ('JM1', 'duplicate')  |             21    |               21 |               21 |         142.6  |           124 |           162 |          0.012 |      nan    |
| ('JM1', 'geo')        |             11.12 |                6 |               17 |           0    |             0 |             0 |          7.787 |       40.4  |
| ('JM1', 'infotheory') |             10.7  |                5 |               15 |           0    |             0 |             0 |          0.594 |      nan    |
| ('JM1', 'none')       |             21    |               21 |               21 |           0    |             0 |             0 |          0.009 |      nan    |
| ('KC1', 'duplicate')  |             21    |               21 |               21 |          32.52 |            26 |            40 |          0.005 |      nan    |
| ('KC1', 'geo')        |             10.26 |                5 |               15 |           0    |             0 |             0 |          1.198 |       40.36 |
| ('KC1', 'infotheory') |             11.8  |                5 |               15 |           0    |             0 |             0 |          0.1   |      nan    |
| ('KC1', 'none')       |             21    |               21 |               21 |           0    |             0 |             0 |          0.002 |      nan    |
| ('PC1', 'duplicate')  |             21    |               21 |               21 |           9.64 |             6 |            12 |          0.004 |      nan    |
| ('PC1', 'geo')        |             10.56 |                5 |               15 |           0    |             0 |             0 |          1.162 |       41.32 |
| ('PC1', 'infotheory') |             12.9  |                5 |               15 |           0    |             0 |             0 |          0.093 |      nan    |
| ('PC1', 'none')       |             21    |               21 |               21 |           0    |             0 |             0 |          0.002 |      nan    |

### Most frequently selected metrics: geo

- CM1: lOCode (60%), t (58%), iv(g) (56%), lOComment (56%), i (54%), v (54%)
- JM1: lOBlank (78%), iv(g) (70%), v(g) (64%), lOCode (64%), l (60%), i (56%)
- KC1: d (64%), lOComment (62%), v(g) (60%), v (58%), total_Op (58%), iv(g) (56%)
- PC1: lOComment (76%), locCodeAndComment (68%), T (64%), N (62%), v(g) (58%), D (56%)

### Most frequently selected metrics: infotheory

- CM1: uniq_Opnd (100%), b (94%), lOComment (92%), v (86%), loc (84%), uniq_Op (56%)
- JM1: loc (100%), b (100%), lOCode (100%), lOBlank (100%), uniq_Opnd (90%), iv(g) (80%)
- KC1: n (100%), v (100%), b (100%), total_Opnd (100%), e (96%), total_Op (94%)
- PC1: loc (100%), I (100%), lOBlank (100%), uniq_Opnd (100%), B (98%), lOCode (94%)

### Cleaning log
```
{
 "CM1": {
  "modules_raw": 498,
  "defective_raw": 49,
  "removed_missing_label": 0,
  "removed_exact_duplicates": 56,
  "modules_clean": 442,
  "defective_clean": 48,
  "defect_rate_clean": 0.1085972850678733,
  "missing_metric_cells": 0,
  "rows_with_missing_metric": 0,
  "inconsistent_records_after_cleaning": 2,
  "n_metrics": 21,
  "file": "cm1.arff",
  "sha256": "fbd4a74687d24350ee90b67211698c15406be0b6e435b98f5039e94302b97f09"
 },
 "JM1": {
  "modules_raw": 10885,
  "defective_raw": 2106,
  "removed_missing_label": 0,
  "removed_exact_duplicates": 1973,
  "modules_clean": 8912,
  "defective_clean": 2007,
  "defect_rate_clean": 0.2252019748653501,
  "missing_metric_cells": 25,
  "rows_with_missing_metric": 5,
  "inconsistent_records_after_cleaning": 176,
  "n_metrics": 21,
  "file": "jm1.arff",
  "sha256": "8ac4744e0807b49158964a05f1066da3a9e7f8fa27f47a52e80031e7942c5d97"
 },
 "KC1": {
  "modules_raw": 2109,
  "defective_raw": 326,
  "removed_missing_label": 0,
  "removed_exact_duplicates": 897,
  "modules_clean": 1212,
  "defective_clean": 315,
  "defect_rate_clean": 0.2599009900990099,
  "missing_metric_cells": 0,
  "rows_with_missing_metric": 0,
  "inconsistent_records_after_cleaning": 40,
  "n_metrics": 21,
  "file": "kc1.arff",
  "sha256": "82fa208e6fc2c3b96072401c11c077897f713d8d93f7ad4cda89b8e4636bb5c0"
 },
 "PC1": {
  "modules_raw": 1109,
  "defective_raw": 77,
  "removed_missing_label": 0,
  "removed_exact_duplicates": 155,
  "modules_clean": 954,
  "defective_clean": 70,
  "defect_rate_clean": 0.07337526205450734,
  "missing_metric_cells": 0,
  "rows_with_missing_metric": 0,
  "inconsistent_records_after_cleaning": 12,
  "n_metrics": 21,
  "file": "pc1.arff",
  "sha256": "b57312981180a748de4fc618e8c1173e01fc9d5c9923048e1aaa2d4572a95b35"
 }
}
```