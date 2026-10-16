# pyright: reportMissingImports=false

import pandas as pd
import numpy as np
from utils.preprocessing import (
    clean_data, encode_categorical, encode_target,
    handle_missing_values, split_and_scale, apply_smote
)

#dataset yükleme
df = pd.read_csv("data/bank-full.csv", sep=';')

print(" İlk 5 Satır:")
print(df.head())

print("\n Veri Seti Bilgisi:")
print(df.info())

print("\n İstatistiksel Özet:")
print(df.describe(include='all').transpose().head(10))

#dataset temizleme
df_clean = clean_data(df)
print("\n🧼 Temizlenmiş veri örneği:")
print(df_clean.head())

#kategorik encode
df_encoded = encode_categorical(df_clean)
print("\n🎛️ One-Hot Encoding sonrası veri:")
print(df_encoded.head())
print(f"Yeni sütun sayısı: {df_encoded.shape[1]}")

#hedef değişkeni dönüştürme
df_final = encode_target(df_encoded)
print("\n🎯 Y sütunu dönüştürüldü:")
print(df_final['y'].value_counts())

#eksik verileri doldurma
df_ready = handle_missing_values(df_final)

#Eğitim ve test setlerini ayır ve ölçekle
X_train, X_test, y_train, y_test, scaler, feature_cols, numeric_cols = split_and_scale(df_ready)

print("\n Eğitim ve test setleri oluşturuldu:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")

#XGBoost Optimizasyonu
from models.xgboost_model import train_xgboost_model, evaluate_xgboost_model

neg, pos = np.bincount(y_train)
scale_weight = round(neg / max(pos, 1), 2)
print(f"\n scale_pos_weight = {scale_weight}")

xgb_model = train_xgboost_model(X_train, y_train, scale_pos_weight=scale_weight)
evaluate_xgboost_model(xgb_model, X_test, y_test)

#SMOTE ile ANN eğitimi için veri dengeleme
from models.ann_model import build_ann_model, evaluate_ann_model
from tensorflow.keras.callbacks import EarlyStopping

X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

#ANN Modeli Oluşturmak
ann_model = build_ann_model(input_dim=X_train_balanced.shape[1])

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
class_weights = {0: 1.0, 1: 1.3}

#ANN Modelini Eğit
ann_model.fit(
    X_train_balanced, y_train_balanced,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

#ANN Modelini değerlendir
evaluate_ann_model(ann_model, X_test, y_test)


# Segmentasyon (KMeans)

from segmentation.segmentation import (
    prepare_for_clustering, silhouette_scores, pick_k_by_silhouette,
    fit_kmeans, pca_scatter, attach_clusters, churn_by_cluster, save_outputs,
    cluster_profile
)

# 1) Ölçeklenmiş X'i hazırla (y'yi çıkar)
X_clust, X_feat_df, scaler_clust = prepare_for_clustering(df_ready, drop_cols=["y"])
# (aşağıda pca_scatter ve kmeans için X_clust'ı kullanmaya devam; 
# transform_all kısmında EĞİTİM scaler'ı olan 'scaler' değişkenini kullanıyoruz)

# 2) k seçimi (otomatik silhouette ile)
scores = silhouette_scores(X_clust, k_min=2, k_max=8)
k = pick_k_by_silhouette(scores)
print(f"\n Auto-picked k by silhouette: k={k}, score={scores[k]:.4f}")

# 3) KMeans
labels, km = fit_kmeans(X_clust, n_clusters=k)

# 4) PCA görselleştirme
pca_scatter(X_clust, labels)

# 5) Raporlar
df_with_clusters = attach_clusters(df_ready, labels, colname="Cluster")
churn_tbl = churn_by_cluster(df_with_clusters, y_col="y", cluster_col="Cluster")
print("\n Segment bazlı churn (%):")
print(churn_tbl)

# 6) Çıktıları klasöre kaydetme
save_outputs(df_with_clusters, churn_tbl, outdir="outputs")

# 7) Cluster profili (sayısal ort., kategorik oranlar)
categorical_cols = ["job", "marital", "education", "housing", "loan", "contact", "month", "poutcome"]
categorical_cols = [c for c in categorical_cols if c in df_with_clusters.columns]

profile_tbl = cluster_profile(df_with_clusters, cluster_col="Cluster", categorical_cols=categorical_cols)
profile_tbl.to_csv("outputs/cluster_profile.csv")

print("\n outputs/segmented.csv, outputs/churn_by_cluster.csv ve outputs/cluster_profile.csv kaydedildi.")


# (7) Model Tahminlerine Göre Segment Bazlı Churn Analizi (XGBoost)

from utils.preprocessing import transform_all

# 1) Eğitimdeki aynı scaler + kolon sırası ile tüm veriyi dönüştür
X_all_scaled = transform_all(
    df_ready, scaler=scaler, numeric_cols=numeric_cols,
    target_col='y', feature_cols=feature_cols
)

# 2) Olasılık tahminleri (pozitif sınıf)
y_pred_proba_all = xgb_model.predict_proba(X_all_scaled)[:, 1]

# 2.1 Threshold uygulama
threshold = 0.637  # optimize edilmiş eşik değeri
y_pred_label_all = (y_pred_proba_all >= threshold).astype(int)

# 3) Tahminleri cluster DF'ine ekler
df_with_clusters["predicted_churn_prob"] = y_pred_proba_all
df_with_clusters["predicted_churn_label"] = y_pred_label_all

# 4) Segment bazında ortalama tahmini churn (%) - threshold sonrası
predicted_churn_by_cluster = (
    df_with_clusters.groupby("Cluster")["predicted_churn_label"]
    .mean().mul(100).round(2)
    .rename("predicted_churn_%")
)

print("\n Model Tahminlerine Göre Segment Bazlı Churn (% - threshold uygulanmış):")
print(predicted_churn_by_cluster)

# 5) Gerçek vs Tahmin karşılaştırması
comparison_tbl = churn_tbl.join(predicted_churn_by_cluster, how="left")
print("\n Gerçek vs Model Tahmini Churn (%):")
print(comparison_tbl)

# 6) Kaydet
comparison_tbl.to_csv("outputs/real_vs_predicted_churn_by_cluster.csv", index=True)
print("\n outputs/real_vs_predicted_churn_by_cluster.csv kaydedildi.")

# --- Segment Bazlı Kampanya Önerileri ---
from models.campaign_recommendations import recommend_for_clusters

recommendations = recommend_for_clusters(profile_tbl, comparison_tbl)

# Kaydet
recommendations.to_csv("outputs/campaign_recommendations.csv", index=False)
recommendations.to_json("outputs/campaign_recommendations.json", orient="records", force_ascii=False)

print("\n Kampanya önerileri kaydedildi: outputs/campaign_recommendations.(csv/json)")

# PDF rapora basit bir sayfa ekleyelim
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pdf_path2 = "outputs/cluster_analysis_report.pdf"
with PdfPages(pdf_path2) as pdf:
    # mevcut pdf'i yeniden oluşturmak yerine, sadece yeni bir sayfa ekliyoruz:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    tbl_text = "Segment-based Campaign Recommendations\n\n"
    head = ["Cluster", "Risk", "Primary_Channel", "Recommended_Offers"]
    rows = recommendations[head].astype(str).values.tolist()
    # Metin olarak yaz (özet)
    lines = [", ".join(head)]
    lines += [", ".join(r) for r in rows]
    tbl_text += "\n".join(lines[:40])  # çok uzunsa ilk 40 satır
    ax.text(0.01, 0.99, tbl_text, va='top', ha='left', wrap=True)
    pdf.savefig(fig); plt.close(fig)

print(f" PDF'e öneri sayfası eklendi: {pdf_path2}")

#Grafik karşılaştırması
import matplotlib.pyplot as plt

# 8. Adım: Görselleştirme - Gerçek vs Tahmin Churn Oranları
plt.figure(figsize=(8, 5))
comparison_tbl[["churn_rate_%", "predicted_churn_%"]].plot(
    kind="bar", figsize=(8, 5)
)
plt.title("Gerçek vs Tahmin Churn Oranları (Segment Bazında)")
plt.xlabel("Cluster")
plt.ylabel("Churn Oranı (%)")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(["Gerçek", "Tahmin"])
plt.tight_layout()
plt.savefig("outputs/churn_comparison.png")  # Görsel kaydedilir
plt.show()


# Segment Profil Görselleri

import matplotlib.pyplot as plt

# 1) Ortalama Yaş
plt.figure(figsize=(6, 4))
profile_tbl["age"].plot(kind="bar", color="skyblue")
plt.title("Segmentlere Göre Ortalama Yaş")
plt.xlabel("Cluster")
plt.ylabel("Ortalama Yaş")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/avg_age_by_cluster.png")
plt.show()

# 2) Ortalama Bakiye
plt.figure(figsize=(6, 4))
profile_tbl["balance"].plot(kind="bar", color="orange")
plt.title("Segmentlere Göre Ortalama Bakiye (€)")
plt.xlabel("Cluster")
plt.ylabel("Ortalama Bakiye (€)")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/avg_balance_by_cluster.png")
plt.show()

# 3) Kategorik Değişken Oranları 
job_cols = [col for col in profile_tbl.columns if col.startswith("job=")]
if job_cols:
    profile_tbl[job_cols].plot(
        kind="bar", stacked=True, figsize=(10, 6), colormap="tab20"
    )
    plt.title("Segmentlere Göre Meslek Dağılımları")
    plt.xlabel("Cluster")
    plt.ylabel("Oran")
    plt.xticks(rotation=0)
    plt.legend(title="Meslek", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/job_distribution_by_cluster.png")
    plt.show()



# Görselleri PDF Raporuna Kaydetme

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pdf_path = "outputs/cluster_analysis_report.pdf"
with PdfPages(pdf_path) as pdf:
    # 1) Gerçek vs Tahmin Churn Oranları
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    comparison_tbl[["churn_rate_%", "predicted_churn_%"]].plot(
        kind="bar", ax=ax1
    )
    ax1.set_title("Gerçek vs Tahmin Churn Oranları (Segment Bazında)")
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Churn Oranı (%)")
    ax1.set_xticks(range(len(comparison_tbl.index)))
    ax1.set_xticklabels(comparison_tbl.index)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend(["Gerçek", "Tahmin"])
    pdf.savefig(fig1)
    plt.close(fig1)

    # 2) Ortalama Yaş
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    profile_tbl["age"].plot(kind="bar", color="skyblue", ax=ax2)
    ax2.set_title("Segmentlere Göre Ortalama Yaş")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Ortalama Yaş")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    pdf.savefig(fig2)
    plt.close(fig2)

    # 3) Ortalama Bakiye
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    profile_tbl["balance"].plot(kind="bar", color="orange", ax=ax3)
    ax3.set_title("Segmentlere Göre Ortalama Bakiye (€)")
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Ortalama Bakiye (€)")
    ax3.grid(axis="y", linestyle="--", alpha=0.7)
    pdf.savefig(fig3)
    plt.close(fig3)

    # 4) Kategorik Değişken Oranları 
    job_cols = [col for col in profile_tbl.columns if col.startswith("job=")]
    if job_cols:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        profile_tbl[job_cols].plot(
            kind="bar", stacked=True, colormap="tab20", ax=ax4
        )
        ax4.set_title("Segmentlere Göre Meslek Dağılımları")
        ax4.set_xlabel("Cluster")
        ax4.set_ylabel("Oran")
        ax4.legend(title="Meslek", bbox_to_anchor=(1.05, 1), loc="upper left")
        pdf.savefig(fig4)
        plt.close(fig4)

print(f"\n PDF rapor oluşturuldu: {pdf_path}")

# --- Kampanya Önerileri PDF'e Ek ---
from models.campaign_recommendations import recommend_for_clusters
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Önerileri yükle
recommendations = recommend_for_clusters(profile_tbl, comparison_tbl)

pdf_path = "outputs/cluster_analysis_report.pdf"
with PdfPages(pdf_path) as pdf:
    # Önce mevcut sayfaları yeniden üretmek yerine
    # sadece ek sayfa olarak öneri tablosunu ekle
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Tablo başlığı
    tbl_text = "Segment-based Campaign Recommendations\n\n"
    headers = ["Cluster", "Risk", "Primary_Channel", "Recommended_Offers", "Profile_Snapshot"]
    rows = recommendations[headers].astype(str).values.tolist()

    # Satırları yaz
    lines = [", ".join(headers)]
    for r in rows:
        lines.append(", ".join(r))

    # Çok uzunsa ilk 20 satır
    tbl_text += "\n".join(lines[:20])

    # PDF'e ekle
    ax.text(0.01, 0.99, tbl_text, va='top', ha='left', wrap=True, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)

print(f"📄 Kampanya önerileri PDF'e eklendi: {pdf_path}")