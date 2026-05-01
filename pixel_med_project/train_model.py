import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap

warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING XGBOOST MODEL")
print("=" * 60)

# Load dataset (CHANGE THIS TO YOUR FILE NAME)
df = pd.read_csv('synthetic_medical_triage.csv')

print(f"\n[1] Loading data... Shape: {df.shape}")
print(f"    Original columns: {list(df.columns)}")

# ============================================================
# REMOVE previous_er_visits COLUMN IF IT EXISTS
# ============================================================

if 'previous_er_visits' in df.columns:
    print(f"\n[2] ⚠️ Found 'previous_er_visits' column - Removing it...")
    df = df.drop('previous_er_visits', axis=1)
    print(f"    ✅ Column 'previous_er_visits' has been removed")
else:
    print(f"\n[2] ✅ 'previous_er_visits' column not found - No action needed")

# Also remove any other unwanted columns (optional)
unwanted_columns = ['patient_id', 'id', 'name', 'timestamp', 'date', 'visit_id']
existing_unwanted = [col for col in unwanted_columns if col in df.columns]
if existing_unwanted:
    print(f"    Also removing: {existing_unwanted}")
    df = df.drop(existing_unwanted, axis=1)

print(f"    New columns: {list(df.columns)}")

# ============================================================
# CLEAN AND PREPARE DATA
# ============================================================

# Remove patients with age <= 0
df = df[df['age'] > 0]

# Encode arrival_mode (text to numbers)
if 'arrival_mode' in df.columns and df['arrival_mode'].dtype == 'object':
    le = LabelEncoder()
    df['arrival_mode'] = le.fit_transform(df['arrival_mode'])
    print(f"    Encoded arrival_mode: walk_in=0, wheelchair=1, ambulance=2")

# Remove missing values
df = df.dropna()
print(f"\n[3] Clean shape: {df.shape}")

# ============================================================
# DEFINE FEATURES (8 features - NO previous_er_visits)
# ============================================================

features = [
    'age',
    'heart_rate',
    'systolic_blood_pressure',
    'oxygen_saturation',
    'body_temperature',
    'pain_level',
    'chronic_disease_count',
    'arrival_mode'
]

# Check if all features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"\n⚠️ Warning: Missing features: {missing_features}")
else:
    print(f"\n[4] ✅ All {len(features)} features found")

target = 'triage_level'

X = df[features]
y = df[target]

print(f"    Features: {features}")
print(f"    Target: {target} (0=Low, 1=Medium, 2=High, 3=Critical)")

# ============================================================
# TRAIN MODEL
# ============================================================

print(f"\n[5] Training XGBoost model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"    Training samples: {len(X_train)}")
print(f"    Testing samples: {len(X_test)}")
print(f"    ✅ Accuracy: {accuracy * 100:.2f}%")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n[6] Feature Importance:")
for i, row in importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"    {row['Feature']:25s}: {row['Importance']:.3f} {bar}")

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'], importance['Importance'], color='#3498db')
plt.xlabel('Importance Score')
plt.title('Feature Importance - XGBoost Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print(f"\n    📊 Feature importance plot saved as 'feature_importance.png'")

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low (0)', 'Medium (1)', 'High (2)', 'Critical (3)'],
            yticklabels=['Low (0)', 'Medium (1)', 'High (2)', 'Critical (3)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print(f"    📊 Confusion matrix saved as 'confusion_matrix.png'")

# ============================================================
# SHAP EXPLAINER
# ============================================================

print(f"\n[7] Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Save model and explainer
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

print(f"\n[8] ✅ Model saved as 'xgboost_model.pkl'")
print(f"    ✅ SHAP explainer saved as 'shap_explainer.pkl'")

# Summary SHAP plot
try:
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    📊 SHAP summary plot saved as 'shap_summary.png'")
except Exception as e:
    print(f"    ⚠️ SHAP plot could not be generated: {e}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\n✅ The model uses ONLY these 8 features:")
for f in features:
    print(f"    • {f}")
print("\n🚀 Run 'python api.py' to start the web application")
print("=" * 60)