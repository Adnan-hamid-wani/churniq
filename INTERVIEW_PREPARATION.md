# ChurnIQ interview preparation

### 1. What exactly is the target?
`Churn` is IBM's Yes/No label for whether the sampled customer left. Training maps Yes to 1 and uses that probability column.

### 2. How did you verify the data?
I used IBM's archived repository, checked that the CSV had 7,043 rows and 21 columns, and pinned its SHA-256. The loader validates fields, known categories, both target classes and unique identifiers.

### 3. Why exclude customerID?
An identifier is primarily for tracking and may encourage memorization without a useful generalizable signal. Batch inference carries it through to the export separately.

### 4. How did you handle TotalCharges?
The CSV has 11 blank entries, all at zero tenure. They become missing values. The numerical imputer learns a median within each fit, including each cross-validation fold.

### 5. How do you distinguish missing data from explicit no-service categories?
Blanks are missing. Values such as No internet service encode eligibility and are kept as ordinary categories.

### 6. What happens to an unseen categorical level?
OneHotEncoder with `handle_unknown="ignore"` creates an all-zero block for that field. The batch UI flags unfamiliar levels so the user can catch typos or changes in the source data.

### 7. Why scale numbers only for Logistic Regression?
Scaling puts numerical values on comparable scales for the regularized regression objective. Trees can make the same ordering-based splits without scaling.

### 8. What stops leakage across folds?
GridSearchCV receives the full pipeline. Each clone fits its transformer and estimator using only its training fold. Validation rows are transformed using those fitted statistics.

### 9. Why stratify the split and folds?
Churn occurs in about 26.5% of the sample. Stratification preserves that proportion approximately across partitions and validation folds.

### 10. Which hyperparameters did you tune?
I varied C for Logistic Regression and maximum depth and minimum leaf sample counts for the tree and forest. Forest size was fixed at 120 for runtime.

### 11. Why use ROC-AUC for model selection?
It measures ranking across cutoffs, which fits the objective of detecting relatively high-risk accounts. I also report average precision and cutoff-based metrics because ranking alone does not tell us how many churners a team will reach.

### 12. Which model won, and how certain are you?
Random Forest led mean validation ROC-AUC at 0.8483 versus 0.8463 for Logistic Regression. The gap is small and this split and fold setup cannot establish a general advantage.

### 13. Why report the held-out comparison for all three models?
It shows practical differences between methods under the same data and cutoff. The model selection criterion is mean validation ROC-AUC, so the test scores are diagnostic.

### 14. What does the confusion matrix say?
At 0.50, the forest correctly flagged 187 churners and missed 187. It falsely flagged 90 customers and correctly predicted Stay for 945.

### 15. Would you use 0.50 with a retention team?
I would ask about contact capacity, the costs of unnecessary contact and missed churn, and the expected benefit of an intervention. I would choose the operating point using validation data and confirm it on a separate test set.

### 16. What is the difference between a risk band and the predicted class?
Risk bands divide the score at 0.30 and 0.70. The predicted class uses 0.50, so a medium-risk customer can receive either class.

### 17. How did you compute global feature importance?
I shuffled each original feature eight times on the test set and measured the mean decrease in ROC-AUC. Contract and tenure ranked highest. Related inputs can dilute or distort their individual importance.

### 18. How does an individual explanation work?
I keep the other features fixed and replace one input with its training median or mode. I report how many percentage points the original score differs from the modified score. It measures the fitted model's sensitivity to this comparison and can contain off-distribution combinations.

### 19. How do you keep training separate from deployment?
`python -m src.train` produces the Joblib model and the metrics JSON. The release includes both. Streamlit caches model loading and runs inference through the same fitted preprocessing, with a version and checksum check.

### 20. What would you test before real-world use?
I would verify that features exist at scoring time, use a later-period holdout, evaluate score calibration and subgroup error patterns, and compare the expected operational impact of several thresholds. I would test whether a retention intervention changes outcomes before using the scores to target it.
