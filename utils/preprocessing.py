import pandas as pd

# ============== Cleaning & Encoding ==============

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 'duration' sütununu çıkar (etiketsiz bilgi)
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
    # 2) NaN
    df = df.replace('unknown', pd.NA)
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # hedef kolonu hariç tüm object kolonları one-hot
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y')
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

def encode_target(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    if target_col in df.columns:
        df[target_col] = (
            df[target_col]
            .map({'yes': 1, 'no': 0})
            .astype('Int64')
            .fillna(0)
            .astype(int)
        )
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\n Eksik Değer Doldurma Başladı...")
    # kategorikler
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna('missing')
    # sayısallar (int, float, bool)
    num_cols = df.select_dtypes(include=['int64', 'float64', 'bool']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    print(" Eksik değerler başarıyla dolduruldu.")
    return df

# ============== Train/Test + Scaling ==============
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_scale(
    df: pd.DataFrame,
    target_col: str = 'y',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Dönüş:
      X_train_scaled, X_test_scaled, y_train, y_test,
      scaler, feature_cols, numeric_cols
    """
    # 1) X, y ayır
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2) böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3) sadece sayısal kolonları ölçekle
    feature_cols = X.columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()

    scaler = StandardScaler()

    # Kopyaları oluştur
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    # DataFrame olarak atama (dtype uyarılarını engellemek içindi) ---
    X_train_scaled[numeric_cols] = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_cols].astype(float)),
        index=X_train.index,
        columns=numeric_cols
    )
    X_test_scaled[numeric_cols] = pd.DataFrame(
        scaler.transform(X_test[numeric_cols].astype(float)),
        index=X_test.index,
        columns=numeric_cols
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, numeric_cols

# ============== SMOTE ==============
''' SMOTE kullanma nedeni: 	
    	- Veri setinde sınıf dengesizliği vardı → Churn eden müşteri sayısı (1 sınıfı) churn etmeyenlere (0 sınıfı) göre çok daha azdı.
		- Bu dengesizlikte model, çoğunluk sınıfını öğrenmeye meyilli olur ve azınlık sınıfını (churn edenleri) iyi tahmin edemez.
		- SMOTE, azınlık sınıfındaki örneklerden sentetik (yeni) örnekler üreterek veri setini dengeler.
		- Böylece model, churn eden müşterilerin özelliklerini daha iyi öğrenir ve Recall (y=1) artar.'''
# pyright: reportMissingImports=false
from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    print("\n SMOTE uygulanıyor...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(" SMOTE sonrası veri boyutları:")
    print(f"X_resampled: {X_resampled.shape}")
    print(f"y_resampled: {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled

# ============== Transform-All (inference) ==============

def transform_all(
    df: pd.DataFrame,
    scaler: StandardScaler,
    numeric_cols,
    target_col: str = 'y',
    feature_cols=None
):
    """
    Eğitimde fit edilen scaler ile TÜM veriyi aynı sırada transform eder.
    feature_cols verilirse kolon sırası eğitimle birebir korunur.
    """
    X_all = df.drop(columns=[target_col]).copy()
    if feature_cols is not None:
        X_all = X_all[feature_cols]

    # DataFrame olarak atama (dtype uyarılarını engellemek içindi) ---
    X_all[numeric_cols] = pd.DataFrame(
        scaler.transform(X_all[numeric_cols].astype(float)),
        index=X_all.index,
        columns=numeric_cols
    )
    return X_all