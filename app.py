# pyright: reportMissingImports=false
import streamlit as st
import pandas as pd
from pathlib import Path

# Altair opsiyonel: yoksa fallback ile bar_chart kullan
try:
    import altair as alt
    FORCE_FALLBACK = False
except Exception:
    alt = None
    FORCE_FALLBACK = True

st.set_page_config(page_title="Bank Churn & Segmentation", layout="wide")

# --- Veri yolları ---
OUT = Path("outputs")
FILES = {
    "recs": OUT / "campaign_recommendations.csv",
    "comp": OUT / "real_vs_predicted_churn_by_cluster.csv",
    "prof": OUT / "cluster_profile.csv",
    "segm": OUT / "segmented.csv",
    "pdf":  OUT / "cluster_analysis_report.pdf",
}

@st.cache_data
def load_csv(p: Path):
    if p.exists():
        return pd.read_csv(p)
    return None

def ensure_cluster_col(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if "Cluster" not in df.columns:
        # sık görülen durumlar: index=Cluster ya da "Unnamed: 0"
        if getattr(df, "index", None) is not None and df.index.name in (None, "Cluster"):
            df = df.reset_index()
        if "Unnamed: 0" in df.columns and "Cluster" not in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Cluster"})
    return df

# --- Yüklemeler ---
recs = load_csv(FILES["recs"])
comp = ensure_cluster_col(load_csv(FILES["comp"]))
prof = ensure_cluster_col(load_csv(FILES["prof"]))
segm = load_csv(FILES["segm"])

st.title(" Customer Retention Dashboard")
st.caption("Churn Prediction • Behavioral Segmentation • Campaign Recommendations")

# --- Genel kartlar ---
col1, col2, col3 = st.columns(3)
if comp is not None and not comp.empty and "predicted_churn_%" in comp.columns:
    avg_pred = comp["predicted_churn_%"].mean()
    col1.metric("Clusters", len(comp))
    col2.metric("Avg Predicted Churn", f"{avg_pred:.1f}%")
else:
    col1.metric("Clusters", "-" if comp is None else len(comp))
    col2.metric("Avg Predicted Churn", "-")

if segm is not None and not segm.empty:
    col3.metric("Segmented Rows", f"{len(segm):,}")
else:
    col3.metric("Segmented Rows", "-")

st.divider()

# --- Cluster seçimi ---
cluster_ids = recs["Cluster"].tolist() if (recs is not None and "Cluster" in recs.columns) else []
selected_cluster = st.selectbox("Select Cluster", cluster_ids, index=0 if cluster_ids else None)

if selected_cluster is not None and recs is not None and comp is not None:
    st.subheader(f" Cluster {selected_cluster} Overview")

    rec_row = recs[recs["Cluster"] == selected_cluster].iloc[0]
    comp_row = comp[comp["Cluster"] == selected_cluster].iloc[0]

    # KPI'lar
    k1, k2, k3 = st.columns(3)
    k1.metric("Risk", str(rec_row.get("Risk", "-")))
    k2.metric("Real Churn", f"{float(comp_row.get('churn_rate_%', 0)):.1f}%")
    k3.metric("Predicted Churn", f"{float(comp_row.get('predicted_churn_%', 0)):.1f}%")

    st.markdown("###  Campaign Recommendation")
    st.write(f"**Primary Channel:** {rec_row.get('Primary_Channel', '-')}")
    st.write(f"**Profile Snapshot:** {rec_row.get('Profile_Snapshot', '-')}")

    st.markdown("**Recommended Offers:**")
    offers = [o.strip() for o in str(rec_row.get("Recommended_Offers", "")).split(",")]
    for o in offers:
        if o:
            st.write(f"- {o}")

    # ----------------------------
    #  Cluster-level Charts
    # ----------------------------
    st.divider()
    st.subheader(" Cluster-level Charts")

    use_altair = (alt is not None) and (not FORCE_FALLBACK)

    # 1) Gerçek vs Tahmin Churn
    if comp is not None and {"churn_rate_%", "predicted_churn_%", "Cluster"}.issubset(comp.columns):
        st.markdown("**Gerçek vs Tahmin Churn Oranları (Segment Bazında)**")
        if use_altair:
            cdf = comp.copy()
            churn_long = cdf.melt(
                id_vars="Cluster",
                value_vars=["churn_rate_%", "predicted_churn_%"],
                var_name="metric",
                value_name="rate",
            )
            chart_churn = (
                alt.Chart(churn_long)
                .mark_bar()
                .encode(
                    x=alt.X("Cluster:O", title="Cluster"),
                    y=alt.Y("rate:Q", title="Churn (%)"),
                    color=alt.Color("metric:N", title=""),
                    tooltip=[
                        "Cluster",
                        alt.Tooltip("metric:N", title="Type"),
                        alt.Tooltip("rate:Q", title="%", format=".1f"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(chart_churn, use_container_width=True)
        else:
            st.bar_chart(comp.set_index("Cluster")[["churn_rate_%", "predicted_churn_%"]])

    # 2) Ortalama Yaş & Bakiye
    if prof is not None:
        pp = prof.copy()
        if "Cluster" in pp.columns:
            pp = pp.set_index("Cluster")
        left, right = st.columns(2)
        if "age" in pp.columns:
            with left:
                st.markdown("**Ortalama Yaş (Segment Bazında)**")
                if use_altair:
                    st.altair_chart(
                        alt.Chart(pp.reset_index()[["Cluster", "age"]])
                        .mark_bar()
                        .encode(
                            x=alt.X("Cluster:O", title="Cluster"),
                            y=alt.Y("age:Q", title="Average Age"),
                            tooltip=["Cluster", alt.Tooltip("age:Q", title="Age", format=".0f")],
                        )
                        .properties(height=280),
                        use_container_width=True,
                    )
                else:
                    st.bar_chart(pp[["age"]])
        if "balance" in pp.columns:
            with right:
                st.markdown("**Ortalama Bakiye (Segment Bazında)**")
                if use_altair:
                    st.altair_chart(
                        alt.Chart(pp.reset_index()[["Cluster", "balance"]])
                        .mark_bar()
                        .encode(
                            x=alt.X("Cluster:O", title="Cluster"),
                            y=alt.Y("balance:Q", title="Average Balance (€)"),
                            tooltip=["Cluster", alt.Tooltip("balance:Q", title="€", format=",.0f")],
                        )
                        .properties(height=280),
                        use_container_width=True,
                    )
                else:
                    st.bar_chart(pp[["balance"]])

st.divider()

# --- Hızlı indir butonları + PDF ---
dl1, dl2, dl3, dl4, dl5 = st.columns(5)
if recs is not None:
    dl1.download_button("⬇️ Recommendations CSV", recs.to_csv(index=False).encode("utf-8"),
                        "campaign_recommendations.csv", "text/csv")
if comp is not None:
    dl2.download_button("⬇️ Churn Table CSV", comp.to_csv(index=False).encode("utf-8"),
                        "real_vs_predicted_churn_by_cluster.csv", "text/csv")
if prof is not None:
    prof_csv = (prof.reset_index() if "Cluster" not in prof.columns and getattr(prof, "index", None) is not None else prof) \
        .to_csv(index=False).encode("utf-8")
    dl3.download_button("⬇️ Cluster Profile CSV", prof_csv, "cluster_profile.csv", "text/csv")
if segm is not None:
    dl4.download_button("⬇️ Segmented CSV", segm.to_csv(index=False).encode("utf-8"),
                        "segmented.csv", "text/csv")

# PDF report download
if FILES["pdf"].exists():
    with open(FILES["pdf"], "rb") as f:
        dl5.download_button("⬇️ Download PDF Report", f, file_name="cluster_analysis_report.pdf", mime="application/pdf")
else:
    dl5.write(" PDF yok")

# --- Tablolar ---
st.subheader(" Full Tables")
tabs = st.tabs(["Recommendations", "Real vs Predicted Churn", "Cluster Profile", "Segmented"])

with tabs[0]:
    if recs is not None and not recs.empty:
        st.dataframe(recs, use_container_width=True)
    else:
        st.warning("No recommendations found.")

with tabs[1]:
    if comp is not None and not comp.empty:
        st.dataframe(comp, use_container_width=True)
    else:
        st.warning("No churn comparison table found.")

with tabs[2]:
    if prof is not None:
        try:
            prof_to_show = prof.reset_index() if getattr(prof, "index", None) is not None else prof
            st.dataframe(prof_to_show, use_container_width=True)
        except Exception:
            st.dataframe(prof, use_container_width=True)
    else:
        st.warning("No cluster profile found.")

with tabs[3]:
    if segm is not None and not segm.empty:
        st.dataframe(segm, use_container_width=True)
    else:
        st.info("Segmented data not found (optional).")