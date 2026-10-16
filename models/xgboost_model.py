# pyright: reportMissingImports=false
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix,
    roc_auc_score, classification_report,
    precision_recall_curve
)
import numpy as np

def train_xgboost_model(X_train, y_train, scale_pos_weight=None):
    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)
    return model

def evaluate_xgboost_model(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Threshold'u optimize et (F1-score'a göre)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_index = np.argmax(f1s)
    best_threshold = thresholds[best_index]

    print(f"\n Optimize Edilmiş Eşik Değeri: {round(best_threshold, 3)}")

    y_pred = (y_pred_prob > best_threshold).astype(int)

    print("\n XGBoost Model Değerlendirme:")
    print("Accuracy       :", round(accuracy_score(y_test, y_pred), 4))
    print("F1 Score       :", round(f1_score(y_test, y_pred), 4))
    print("Recall (y=1)   :", round(recall_score(y_test, y_pred), 4))
    print("Precision (y=1):", round(precision_score(y_test, y_pred), 4))
    print("ROC AUC Score  :", round(roc_auc_score(y_test, y_pred_prob), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))