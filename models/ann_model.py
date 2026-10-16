# pyright: reportMissingImports=false
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

def build_ann_model(input_dim):
    
    model = Sequential([
        Input(shape=(input_dim,)),         # İlk katman: Input
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")     # Çıkış katmanı (binary classification)
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_ann_model(model, X_test, y_test):
    """
    Verilen ANN modelini test verisinde değerlendirir.
    """
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print("\n ANN Model Değerlendirme Sonuçları:")
    print(f"Accuracy       : {acc:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Recall (y=1)   : {recall:.4f}")
    print(f"Precision (y=1): {precision:.4f}")
    print(f"ROC AUC Score  : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))