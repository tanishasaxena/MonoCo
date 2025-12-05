# Regular
Loaded UCI Heart Disease dataset with 920 rows and 16 columns.
Loaded UCI Heart Disease Concept dataset with 920 rows and 21 columns.

Datasets loaded successfully. Datasets have same number of rows.
uci features shape:  (920, 14)  uci labels shape:  (920, 21)
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
Generated features shape:  (100, 14)
Generated labels shape:  (100, 21)
generated labels:  [[0.  0.  0.  ... 0.9 0.1 0. ]
 [0.  0.7 0.7 ... 0.  0.  3. ]
 [0.  1.  1.  ... 0.  0.  4. ]
 ...
 [0.  0.  0.  ... 1.  1.  0. ]
 [0.  1.  1.  ... 0.  0.  4. ]
 [0.  0.  0.  ... 0.  0.  1. ]]
uci features shape:  (920, 14)  uci labels shape:  (920, 21)
column headers before: 
 [0.  0.7 0.8 0.9 0.9 0.4 0.3 0.9 0.9 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  2. ]
Augmenting training data with generated samples...
train features shape:  (836, 14)  train labels shape:  (836, 21)
train labels after: 
 [4. 0. 3. 0. 0. 0. 0. 1. 2. 1. 0. 3. 0. 0. 3. 1. 0. 2. 1. 0. 0. 0. 0. 2.
 0. 1. 1. 1. 0. 4. 4. 1. 1. 0. 1. 0. 2. 3. 0. 3. 0. 0. 2. 2. 0. 3. 3. 1.
 0. 0. 1. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 1. 1. 2. 2. 0. 2. 1. 3. 1.
 0. 1. 1. 0. 0. 1. 0. 4. 3. 0. 3. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 4. 0.
 0. 0. 2. 3. 1. 0. 3. 1. 1. 0. 4. 2. 1. 1. 1. 0. 0. 0. 1. 0. 0. 0. 1. 2.
 0. 1. 0. 1. 0. 0. 3. 1. 0. 1. 3. 2. 1. 1. 0. 0. 3. 1. 0. 0. 0. 0. 0. 3.
 2. 1. 1. 1. 1. 1. 4. 0. 0. 1. 0. 0. 0. 1. 4. 2. 0. 0. 2. 0. 3. 1. 3. 3.
 0. 2. 4. 1. 2. 1. 2. 0. 0. 1. 3. 1. 2. 0. 0. 1. 3. 0. 4. 0. 1. 0. 3. 1.
 0. 0. 1. 2. 1. 1. 3. 2. 2. 0. 1. 0. 0. 2. 1. 0. 1. 1. 1. 0. 0. 0. 1. 0.
 0. 0. 0. 2. 1. 1. 0. 0. 1. 1. 3. 1. 0. 1. 0. 2. 0. 0. 2. 0. 1. 0. 0. 0.
 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 1. 0. 0. 2. 1.
 0. 0. 1. 0. 1. 0. 0. 2. 1. 0. 0. 1. 4. 0. 0. 1. 1. 0. 2. 0. 1. 1. 0. 3.
 1. 2. 0. 2. 0. 0. 3. 0. 0. 0. 0. 1. 0. 0. 1. 3. 1. 2. 0. 0. 3. 3. 1. 1.
 1. 1. 0. 1. 0. 2. 1. 2. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 3. 1. 1. 3. 0. 0.
 2. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 2. 4. 0. 0. 1. 2. 3. 1.
 2. 3. 1. 3. 1. 3. 0. 1. 0. 3. 0. 0. 1. 3. 0. 0. 1. 3. 1. 2. 1. 1. 1. 1.
 1. 0. 4. 1. 1. 0. 0. 1. 1. 0. 0. 3. 1. 2. 0. 0. 0. 0. 0. 0. 0. 1. 0. 2.
 1. 2. 0. 0. 2. 0. 1. 0. 4. 3. 1. 1. 3. 1. 1. 0. 0. 0. 2. 0. 2. 1. 0. 1.
 0. 0. 0. 1. 2. 0. 0. 0. 1. 3. 1. 1. 2. 3. 3. 0. 0. 0. 3. 1. 3. 2. 2. 0.
 2. 1. 0. 0. 1. 1. 0. 1. 3. 0. 0. 4. 3. 0. 1. 0. 0. 0. 0. 3. 2. 2. 1. 1.
 0. 1. 0. 2. 0. 0. 0. 0. 0. 2. 0. 1. 1. 0. 3. 0. 0. 1. 2. 0. 1. 1. 0. 1.
 3. 2. 1. 0. 3. 1. 2. 3. 0. 1. 0. 1. 1. 2. 2. 2. 0. 0. 0. 0. 2. 1. 0. 0.
 0. 1. 3. 0. 0. 4. 2. 3. 0. 0. 1. 2. 3. 0. 0. 1. 1. 2. 3. 0. 2. 2. 2. 3.
 1. 2. 1. 0. 0. 3. 0. 1. 0. 1. 3. 2. 3. 3. 0. 1. 1. 1. 0. 3. 1. 1. 0. 1.
 4. 3. 0. 0. 0. 0. 0. 4. 1. 4. 0. 1. 0. 3. 0. 0. 1. 0. 1. 4. 1. 1. 0. 0.
 0. 3. 0. 0. 2. 0. 0. 0. 1. 0. 1. 1. 1. 1. 0. 1. 0. 4. 1. 2. 0. 1. 4. 1.
 0. 2. 0. 0. 2. 2. 0. 1. 1. 0. 0. 3. 1. 3. 0. 0. 0. 3. 0. 0. 4. 1. 1. 0.
 1. 0. 3. 0. 0. 1. 0. 2. 2. 1. 0. 3. 0. 2. 1. 0. 0. 0. 3. 0. 1. 0. 2. 0.
 0. 0. 1. 0. 0. 0. 2. 3. 3. 1. 0. 1. 1. 0. 2. 1. 1. 1. 2. 1. 3. 0. 1. 1.
 0. 0. 1. 3. 1. 0. 0. 0. 0. 0. 0. 3. 0. 2. 0. 1. 0. 0. 0. 3. 0. 3. 3. 0.
 0. 0. 0. 0. 1. 0. 4. 0. 0. 1. 3. 2. 1. 0. 0. 0. 0. 3. 4. 0. 3. 1. 0. 4.
 1. 2. 0. 4. 0. 1. 3. 0. 4. 1. 3. 0. 2. 0. 4. 1. 3. 1. 4. 0. 2. 3. 1. 0.
 3. 1. 4. 0. 3. 1. 3. 0. 4. 1. 2. 3. 0. 4. 1. 3. 0. 4. 0. 1. 3. 0. 4. 1.
 3. 0. 4. 1. 2. 3. 0. 3. 2. 1. 4. 0. 2. 3. 0. 4. 1. 3. 1. 4. 3. 0. 1. 4.
 0. 0. 3. 2. 0. 1. 4. 0. 4. 0. 2. 3. 0. 4. 0. 0. 3. 0. 4. 1.]
Starting training...
Validation Variance: 0.1334
Validation MSE: 0.0463
Validation Accuracy (within 0.3 for non-zero vals): 0.7808
Validation Accuracy (within 0.3 for all vals): 0.8707
Training complete.

# Augmented + Generated

Loaded UCI Heart Disease dataset with 920 rows and 16 columns.
Loaded generated data with 100 rows and 16 columns.
Loaded UCI Heart Disease Concept dataset with 920 rows and 21 columns.
Loaded generated data with 100 rows and 21 columns.

Datasets loaded successfully. Datasets have same number of rows.
Generated features shape:  (100, 33)
Generated labels shape:  (100, 21)
generated labels:  [[0.  0.  0.  ... 0.9 0.1 0. ]
 [0.  0.7 0.7 ... 0.  0.  3. ]
 [0.  1.  1.  ... 0.  0.  4. ]
 ...
 [0.  0.  0.  ... 1.  1.  0. ]
 [0.  1.  1.  ... 0.  0.  4. ]
 [0.  0.  0.  ... 0.  0.  1. ]]
uci features shape:  (920, 33)  uci labels shape:  (920, 21)
column headers before: 
 [0.  0.7 0.8 0.9 0.9 0.4 0.3 0.9 0.9 0.8 0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  2. ]
Augmenting training data with generated samples...
train features shape:  (836, 33)  train labels shape:  (836, 21)
train labels after: 
 [4. 0. 3. 0. 0. 0. 0. 1. 2. 1. 0. 3. 0. 0. 3. 1. 0. 2. 1. 0. 0. 0. 0. 2.
 0. 1. 1. 1. 0. 4. 4. 1. 1. 0. 1. 0. 2. 3. 0. 3. 0. 0. 2. 2. 0. 3. 3. 1.
 0. 0. 1. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 1. 1. 2. 2. 0. 2. 1. 3. 1.
 0. 1. 1. 0. 0. 1. 0. 4. 3. 0. 3. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 4. 0.
 0. 0. 2. 3. 1. 0. 3. 1. 1. 0. 4. 2. 1. 1. 1. 0. 0. 0. 1. 0. 0. 0. 1. 2.
 0. 1. 0. 1. 0. 0. 3. 1. 0. 1. 3. 2. 1. 1. 0. 0. 3. 1. 0. 0. 0. 0. 0. 3.
 2. 1. 1. 1. 1. 1. 4. 0. 0. 1. 0. 0. 0. 1. 4. 2. 0. 0. 2. 0. 3. 1. 3. 3.
 0. 2. 4. 1. 2. 1. 2. 0. 0. 1. 3. 1. 2. 0. 0. 1. 3. 0. 4. 0. 1. 0. 3. 1.
 0. 0. 1. 2. 1. 1. 3. 2. 2. 0. 1. 0. 0. 2. 1. 0. 1. 1. 1. 0. 0. 0. 1. 0.
 0. 0. 0. 2. 1. 1. 0. 0. 1. 1. 3. 1. 0. 1. 0. 2. 0. 0. 2. 0. 1. 0. 0. 0.
 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 1. 0. 0. 2. 1.
 0. 0. 1. 0. 1. 0. 0. 2. 1. 0. 0. 1. 4. 0. 0. 1. 1. 0. 2. 0. 1. 1. 0. 3.
 1. 2. 0. 2. 0. 0. 3. 0. 0. 0. 0. 1. 0. 0. 1. 3. 1. 2. 0. 0. 3. 3. 1. 1.
 1. 1. 0. 1. 0. 2. 1. 2. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 3. 1. 1. 3. 0. 0.
 2. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 2. 4. 0. 0. 1. 2. 3. 1.
 2. 3. 1. 3. 1. 3. 0. 1. 0. 3. 0. 0. 1. 3. 0. 0. 1. 3. 1. 2. 1. 1. 1. 1.
 1. 0. 4. 1. 1. 0. 0. 1. 1. 0. 0. 3. 1. 2. 0. 0. 0. 0. 0. 0. 0. 1. 0. 2.
 1. 2. 0. 0. 2. 0. 1. 0. 4. 3. 1. 1. 3. 1. 1. 0. 0. 0. 2. 0. 2. 1. 0. 1.
 0. 0. 0. 1. 2. 0. 0. 0. 1. 3. 1. 1. 2. 3. 3. 0. 0. 0. 3. 1. 3. 2. 2. 0.
 2. 1. 0. 0. 1. 1. 0. 1. 3. 0. 0. 4. 3. 0. 1. 0. 0. 0. 0. 3. 2. 2. 1. 1.
 0. 1. 0. 2. 0. 0. 0. 0. 0. 2. 0. 1. 1. 0. 3. 0. 0. 1. 2. 0. 1. 1. 0. 1.
 3. 2. 1. 0. 3. 1. 2. 3. 0. 1. 0. 1. 1. 2. 2. 2. 0. 0. 0. 0. 2. 1. 0. 0.
 0. 1. 3. 0. 0. 4. 2. 3. 0. 0. 1. 2. 3. 0. 0. 1. 1. 2. 3. 0. 2. 2. 2. 3.
 1. 2. 1. 0. 0. 3. 0. 1. 0. 1. 3. 2. 3. 3. 0. 1. 1. 1. 0. 3. 1. 1. 0. 1.
 4. 3. 0. 0. 0. 0. 0. 4. 1. 4. 0. 1. 0. 3. 0. 0. 1. 0. 1. 4. 1. 1. 0. 0.
 0. 3. 0. 0. 2. 0. 0. 0. 1. 0. 1. 1. 1. 1. 0. 1. 0. 4. 1. 2. 0. 1. 4. 1.
 0. 2. 0. 0. 2. 2. 0. 1. 1. 0. 0. 3. 1. 3. 0. 0. 0. 3. 0. 0. 4. 1. 1. 0.
 1. 0. 3. 0. 0. 1. 0. 2. 2. 1. 0. 3. 0. 2. 1. 0. 0. 0. 3. 0. 1. 0. 2. 0.
 0. 0. 1. 0. 0. 0. 2. 3. 3. 1. 0. 1. 1. 0. 2. 1. 1. 1. 2. 1. 3. 0. 1. 1.
 0. 0. 1. 3. 1. 0. 0. 0. 0. 0. 0. 3. 0. 2. 0. 1. 0. 0. 0. 3. 0. 3. 3. 0.
 0. 0. 0. 0. 1. 0. 4. 0. 0. 1. 3. 2. 1. 0. 0. 0. 0. 3. 4. 0. 3. 1. 0. 4.
 1. 2. 0. 4. 0. 1. 3. 0. 4. 1. 3. 0. 2. 0. 4. 1. 3. 1. 4. 0. 2. 3. 1. 0.
 3. 1. 4. 0. 3. 1. 3. 0. 4. 1. 2. 3. 0. 4. 1. 3. 0. 4. 0. 1. 3. 0. 4. 1.
 3. 0. 4. 1. 2. 3. 0. 3. 2. 1. 4. 0. 2. 3. 0. 4. 1. 3. 1. 4. 3. 0. 1. 4.
 0. 0. 3. 2. 0. 1. 4. 0. 4. 0. 2. 3. 0. 4. 0. 0. 3. 0. 4. 1.]
Starting training...
Validation Variance: 0.1334
Validation MSE: 0.0467
Validation Accuracy (within 0.3 for non-zero vals): 0.7617
Validation Accuracy (within 0.3 for all vals): 0.8625
Training complete.