# Customer Churn Prediction - Documentation

## 1. My Approach & Thought Process
To build an effective churn prediction model, I followed a structured process:

### Step 1: Understanding the Data
- The dataset consists of **66 columns**, including:
  - `user_account_id` (unique customer identifier)
  - `user_lifetime` (number of days since account creation)
  - `user_spendings` (total spending in the last month)
  - `user_no_outgoing_activity_in_days` (number of consecutive days with no outgoing activity)
  - `reloads_sum` (total amount reloaded into the account)
  - **Target variable: `churn` (0 = stayed, 1 = churned)**

- **Class Distribution**:
  - 79% of the customers remained active (`churn = 0`).
  - 21% of the customers churned (`churn = 1`).
  - Since the dataset is **imbalanced**, I considered balancing techniques.

- **Missing Values**:  
  - No missing values were found, so no imputation was necessary.

### Step 2: Feature Engineering & Data Preprocessing
- I removed **irrelevant features** such as `user_account_id`, `year`, and `month`, as they do not contribute to churn prediction.
- To ensure all numerical features were comparable, I **standardized the dataset** using `StandardScaler()`.
- The dataset was split into **80% training data and 20% test data** to evaluate model performance.

### Step 3: Feature Importance Analysis
To better understand which factors contribute to churn, I analyzed feature correlations:
- **Strong positive correlations with churn** (higher values indicate higher churn likelihood):
  - `user_no_outgoing_activity_in_days` (number of days without activity)
  - `user_account_balance_last` (remaining account balance)
  - `reloads_sum` (total amount reloaded)

- **Strong negative correlations with churn** (higher values indicate lower churn likelihood):
  - `user_spendings` (monthly spending)
  - `calls_outgoing_count` (number of outgoing calls)
  - `last_100_sms_outgoing_count` (number of outgoing SMS)

This indicates that **inactive customers with high balances and fewer outgoing calls/messages are more likely to churn**.

### Step 4: Model Selection & Training
I tested two machine learning models:
1. **Logistic Regression** (as a baseline)
2. **Random Forest** (for a more powerful approach)

The first model, **Logistic Regression**, achieved:
- **Accuracy**: 82.26%
- **Precision (Churn)**: 55.23%
- **Recall (Churn)**: 80.16%
- **F1-Score (Churn)**: 65.40%
- **ROC-AUC Score**: 88.14%

The recall was high, meaning it correctly identified most churn customers, but the precision was relatively low, leading to more false positives.

I then trained a **Random Forest model**, which initially performed better in terms of precision but had a lower recall. To improve it, I optimized the model using:
- `n_estimators=200` (more trees for stability)
- `max_depth=15` (limiting tree depth to avoid overfitting)
- `min_samples_leaf=10` (ensuring leaves are not too small)
- `class_weight="balanced"` (handling the imbalance in churn classes)

After optimization, **the final model achieved**:
- **Accuracy**: 85.45%
- **Precision (Churn)**: 62.12%
- **Recall (Churn)**: 78.01%
- **F1-Score (Churn)**: 69.16%
- **ROC-AUC Score**: 90.36%

The **optimized Random Forest model had a better balance between precision and recall**, making it more reliable for predicting customer churn.

---

## 2. Challenges & Considerations
- **Imbalanced Dataset**: Since churned customers made up only 21% of the dataset, I used **class weighting** to ensure they were properly detected.
- **Trade-off Between Precision & Recall**: Higher recall ensures more churn customers are caught, but lowers precision, leading to more false alarms.
- **Feature Scaling**: Standardizing numerical features helped models perform better.

---

## 3. Future Improvements
Although the final model performed well, there are some areas that could be further optimized:
- **Hyperparameter tuning using GridSearchCV**: Further refining parameters like `max_features` and `min_samples_split`.
- **Trying XGBoost or LightGBM**: More advanced tree-based models could improve performance.
- **Adding customer behavioral insights**: Including more time-series features like trends in spending over months.

---

## 4. Conclusion
I successfully developed a **customer churn prediction model** that:
*Uses meaningful customer activity features* 
*Employs a Random Forest classifier with tuned parameters*
*Achieves an ROC-AUC score of 90.36%*

This model can help businesses **identify at-risk customers** and take **preventive actions to reduce churn**.
