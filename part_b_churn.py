import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Function for model evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Trains a model and evaluates its performance."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Load dataset
churn_file_path = "datasets/churn_train.csv"
df_churn = pd.read_csv(churn_file_path)

# Check target variable distribution
churn_distribution = df_churn["churn"].value_counts(normalize=True) * 100
print("\nChurn Distribution (%):")
print(churn_distribution.to_string())

# Check for missing values
missing_values = df_churn.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    print("\nMissing Values in Dataset:")
    print(missing_values)
else:
    print("\nNo missing values found in the dataset.")

# Compute feature correlations with churn
correlation_matrix = df_churn.corr()
churn_correlation = correlation_matrix["churn"].sort_values(ascending=False)

# Display only top 10 most correlated features
top_corr = churn_correlation.head(10)

# Visualize feature correlations with churn
plt.figure(figsize=(8, 5))
sns.heatmap(top_corr.to_frame(), annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Top 10 Feature Correlations with Churn")
plt.show()

# Drop irrelevant features (account ID, year, month)
features_to_drop = ["user_account_id", "year", "month"]
df_churn_cleaned = df_churn.drop(columns=features_to_drop)

# Define features (X) and target variable (y)
X = df_churn_cleaned.drop(columns=["churn"])
y = df_churn_cleaned["churn"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (standardization for more stable models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print dataset shapes after preprocessing
print("\nFinal Dataset Shapes:")
print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Optimized Random Forest
rf_optimized = RandomForestClassifier(
    n_estimators=200,  # More trees for more stable forecasts
    max_depth=15,  # Limited depth to avoid overfitting
    min_samples_leaf=10,  # Minimum number of samples per leaf
    class_weight="balanced",  # Balance between churn/non-churn
    random_state=42,
    n_jobs=-1  # Parallel calculation for acceleration
)

# Train and evaluate model
evaluate_model(rf_optimized, X_train_scaled, X_test_scaled, y_train, y_test)


# All Terminal Outputs for Part B:
#
# Churn Distribution (%):
# churn
# 0    79.085
# 1    20.915
#
# No missing values found in the dataset.
#
# Final Dataset Shapes:
# Training set: (48000, 62)
# Test set: (12000, 62)
#
# Model: RandomForestClassifier
# Accuracy: 0.8545
# Precision: 0.6212
# Recall: 0.7801
# F1-Score: 0.6916
# ROC-AUC Score: 0.9036
#
# Classification Report:
#                precision    recall  f1-score   support
#
#            0       0.94      0.87      0.90      9490
#            1       0.62      0.78      0.69      2510
#
#     accuracy                           0.85     12000
#    macro avg       0.78      0.83      0.80     12000
# weighted avg       0.87      0.85      0.86     12000
