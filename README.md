Mallorn-Astronomical-Classification-LGBM

This repository presents a LightGBM-based binary classification pipeline developed for the ''Mallorn Astronomical Classification Challenge'' on Kaggle.  
The goal is to detect tidal disruption events (TDEs) from simulated LSST light curve data.

Problem Overview
Tidal disruption events are extremely rare astronomical phenomena in which a star is torn apart by a supermassive black hole.  
The dataset is highly **imbalanced** and contains noisy, irregularly sampled time-series observations, making classical classification approaches ineffective.

Data & Feature Engineering
Raw light curve observations were transformed into structured features using statistical and variability-based descriptors, including.

Flux statistics (mean, std, min, max)
Variability and dispersion metrics
Time-domain aggregation features
Log-transformed and normalized representations These engineered features enable tree-based models to capture non-linear temporal patterns.

Model Selection Rationale
Different model families were conceptually evaluated based on their bias–variance tradeoff

Linear and polynomial models underfit the complex signal
KNN and deep neural networks tend to overfit sparse and noisy data
Tree-based ensemble methods provide the most robust balance
''LightGBM** was selected due to its ability to model non-linear interactions while remaining stable under class imbalance.

Training & Validation Strategy
Stratified cross-validation was used
**Out-of-Fold (OOF) predictions** were generated to prevent data leakage
Class imbalance was handled via weighted loss functions
OOF predictions were later reused for threshold optimization.

Threshold Optimization (F1)
Instead of using the default 0.5 decision threshold, the classification cutoff was optimized directly on **OOF predictions** to maximize the **F1-score**.
Optimal threshold found: **0.1305**
This significantly improved recall for rare TDE events

Results
**Public Kaggle Leaderboard F1-score:** **0.1305**
The solution prioritizes generalization over leaderboard-specific heuristics

Repository Structure

text
├── feature_engineering.py
├── train_lgb_oof.py
├── optimize_threshold.py
├── plot_threshold_f1.py
├── submission_01305.csv
├── images/
│   └── threshold_f1_curve.png
└── README.md
