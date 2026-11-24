# How ChurnIQ works

## 1. Customer churn

A customer churns when they stop using the provider's service. Each row in this IBM sample describes a customer and records whether they left. The overview measures the sample's observed churn as the fraction of rows marked Yes.

## 2. Binary classification

The target has two possible classes. The pipeline learns from labeled examples and produces a score for the positive class, churn. A fixed cutoff turns that score into a class prediction.

## 3. Why these three models

Logistic Regression models the log-odds of churn as a weighted sum of transformed features. It gives a strong, relatively simple baseline. A Decision Tree partitions customers using feature rules, capturing nonlinear relationships. A Random Forest averages many trees fitted with row and feature sampling, which can reduce the instability of a single tree.

## 4. Preprocessing

ChurnIQ validates the source schema, separates the target and excludes customer IDs. Blank numerical fields become missing values and are imputed with training medians. Blank categorical cells use training modes. One-hot encoding represents each category with an indicator and tolerates new categories. Logistic Regression scales the numerical features for optimization and regularization. Trees use numerical split thresholds and keep the original scale.

## 5. Leakage prevention

Leakage means giving the model information that would be unavailable at the time of prediction or letting evaluation data influence fitting. Every imputation, scaling and encoding step lives inside the pipeline fitted by the grid search. Each fold learns statistics only from its fitting rows. Holdout rows and churn labels never enter the feature transformer. The source provides no timestamps to independently verify future feature availability.

## 6. Train and test split

An 80/20 split with seed 42 reserves 1,409 rows and keeps the fraction of churners similar in each partition. The holdout estimates performance for unseen rows from this sample. It represents a random split, with no temporal separation.

## 7. Cross-validation

Five stratified folds let each training row act as validation data once per parameter setting. Mean validation ROC-AUC ranks the settings. The winning estimator for each model is then refitted on all 5,634 training rows.

## 8. Evaluation metrics

Accuracy counts correct predictions overall. Precision answers how many flagged customers actually left. Recall answers how many churners were caught. F1 balances precision and recall through their harmonic mean. ROC-AUC measures pairwise ranking quality across thresholds. The precision–recall curve describes the tradeoff among positive predictions as the cutoff changes. Average precision summarizes that curve using increments in recall. The confusion matrix gives counts of each actual and predicted class.

## 9. Selection

Random Forest had the highest mean five-fold ROC-AUC, 0.8483, compared with 0.8463 for Logistic Regression. The narrow margin is specific to this experiment. The forest scored 0.8438 test ROC-AUC and 0.6523 average precision. At the fixed 0.50 cutoff, it recalled 50% of churners, while Logistic Regression had higher recall and F1.

## 10. Predictions and risk

Input columns are validated and passed through the fitted preprocessing and model together. Scores at least 0.50 produce Churn. Low, medium and high risk denote probabilities below 0.30, from 0.30 to below 0.70, and at least 0.70. The class cutoff and risk bands serve two different displays. The score represents a statistical prediction with uncertainty and has not been calibrated.

## 11. Explaining the model

On the holdout, permutation importance shuffles one input field at a time. A large fall in ROC-AUC indicates that the pipeline relied on that field for ranking. For a specific customer, ChurnIQ replaces one field with its training median or mode, re-scores the row, and reports the signed change from that reference. Related fields may lead to implausible combinations. Sensitivities can overlap and need not sum to the original score. Both views describe model behavior and neither demonstrates causation.

## 12. Streamlit integration

The entrypoint registers the five pages. Cached data feeds the descriptive pages. Cached Joblib loading reuses the inference pipeline across sessions. Form and batch state stay within a visitor session. Both paths rely on the shared validation and saved preprocessing. The performance page renders the metrics JSON produced by training.

## 13. Limitations

The dataset represents one historical example without a verified prediction horizon or external evaluation. Scores may drift with customer mix, pricing or product changes. The demo uses demographic fields without a subgroup fairness audit. Sample correlations may reflect selection effects, and retention workflows require evidence that an offer changes outcomes.

## 14. Future work

Collect timestamped data and define the prediction horizon. Reserve a later period for testing. Evaluate score calibration, compare subgroup error rates, and select a threshold from training or validation data using retention costs and capacity. Monitor drift after release. To estimate the effect of a retention offer, use a randomized experiment or another defensible causal design.
