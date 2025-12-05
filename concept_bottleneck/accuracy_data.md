# Regular
Loaded UCI Heart Disease dataset with 920 rows and 16 columns.
Loaded UCI Heart Disease Concept dataset with 920 rows and 21 columns.

Datasets loaded successfully. Datasets have same number of rows.
uci features shape:  (920, 14)  uci labels shape:  (920, 21)
column headers before: 
 [0.  0.7 0.8 0.9 0.9 0.4 0.3 0.9 0.9 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  2. ]
column headers after: 
 [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  0.2 0.  0.  0.
 0.  0. ]
Starting training...
Validation Variance: 0.1334
Validation MSE: 0.0454
Validation Accuracy (within 0.3 for non-zero vals): 0.7842
Validation Accuracy (within 0.3 for all vals): 0.8723
Training complete.

# Augmented
Loaded UCI Heart Disease dataset with 920 rows and 16 columns.
Loaded UCI Heart Disease Concept dataset with 920 rows and 21 columns.

Datasets loaded successfully. Datasets have same number of rows.
uci features shape:  (920, 33)  uci labels shape:  (920, 21)
column headers before: 
 [0.  0.7 0.8 0.9 0.9 0.4 0.3 0.9 0.9 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  2. ]
column headers after: 
 [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  0.2 0.  0.  0.
 0.  0. ]
Starting training...
Validation Variance: 0.1334
Validation MSE: 0.0460
Validation Accuracy (within 0.3 for non-zero vals): 0.7600
Validation Accuracy (within 0.3 for all vals): 0.8636
Training complete.


# Generated
Loaded UCI Heart Disease dataset with 920 rows and 16 columns.
Loaded generated data with 100 rows and 16 columns.
Loaded UCI Heart Disease Concept dataset with 920 rows and 21 columns.
Loaded generated data with 100 rows and 21 columns.

Datasets loaded successfully. Datasets have same number of rows.
uci features shape:  (1020, 14)  uci labels shape:  (1020, 20)
column headers before: 
 [0.  0.7 0.8 0.9 0.9 0.4 0.3 0.9 0.9 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0.  0. ]
column headers after: 
 [0.4 0.2 0.6 1.  0.8 1.  0.3 0.6 0.7 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0. ]
Starting training...
Validation Variance: 0.1381
Validation MSE: 0.0565
Validation Accuracy (within 0.3 for non-zero vals): 0.7733
Validation Accuracy (within 0.3 for all vals): 0.8529
Training complete.

# Augmented + Generated

Loaded UCI Heart Disease dataset with 920 rows and 16 columns.
Loaded generated data with 100 rows and 16 columns.
Loaded UCI Heart Disease Concept dataset with 920 rows and 21 columns.
Loaded generated data with 100 rows and 21 columns.

Datasets loaded successfully. Datasets have same number of rows.
uci features shape:  (1020, 33)  uci labels shape:  (1020, 20)
column headers before: 
 [0.  0.7 0.8 0.9 0.9 0.4 0.3 0.9 0.9 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0.  0. ]
column headers after: 
 [0.4 0.2 0.6 1.  0.8 1.  0.3 0.6 0.7 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0. ]
Starting training...
Validation Variance: 0.1381
Validation MSE: 0.0578
Validation Accuracy (within 0.3 for non-zero vals): 0.7567
Validation Accuracy (within 0.3 for all vals): 0.8393
Training complete.