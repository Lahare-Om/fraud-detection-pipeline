import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/creditcard.csv')

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC (average precision):", average_precision_score(y_test, y_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# MODEL A - Logistic Regression with class weighing
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
evaluate("Logistic Regression (class_weight='balanced')", log_reg, X_test, y_test)


# MODEL B - Random Forest with class weighting
rf = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
evaluate("Random Forest (class_weight=balanced)", rf, X_test, y_test)


# MODEL C - Random Forest + SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_sm, y_train_sm)
evaluate("Random Forest + SMOTE", rf_smote, X_test, y_test)


joblib.dump(rf, 'models/fraud_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
