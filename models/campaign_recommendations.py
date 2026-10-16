# pyright: reportMissingImports=false
import pandas as pd
from typing import Dict, Optional, Tuple

# -------------------------
# Risk hesaplama (data-driven)
# -------------------------

def _extract_scores(row: pd.Series) -> Tuple[float, float, float]:
    real_ = float(row.get("churn_rate_%", 0) or 0)
    pred_ = float(row.get("predicted_churn_%", 0) or 0)
    return real_, pred_, max(real_, pred_)


def _compute_data_thresholds(comp: pd.DataFrame) -> Dict[str, float]:
    scores = []
    for _, r in comp.iterrows():
        _, _, mx = _extract_scores(r)
        scores.append(mx)
    s = pd.Series(scores, dtype="float64")
    return {"mean": float(s.mean()), "std": float(s.std(ddof=0))}


def _risk_level_data(score: float, stats: Dict[str, float]) -> str:
    mean, std = stats["mean"], stats["std"]
    if score >= mean + std:
        return "high"
    if score >= mean:
        return "medium"
    return "low"


# -------------------------
# Kanal seçimi
# -------------------------

def _primary_channel(prof_row: pd.Series) -> str:
    channel = "mobile push"  # varsayılan

    # yaş
    age_val = prof_row.get("age", None)
    try:
        age = float(age_val) if age_val is not None else None
    except Exception:
        age = None

    if age is not None:
        if age < 30:
            channel = "mobile push"
        elif age < 40:
            channel = "mobile push"
        elif age < 55:
            channel = "email"
        else:
            channel = "phone/branch"

    # contact (varsa)
    cell_ratio = prof_row.get("contact=cellular")
    tel_ratio = prof_row.get("contact=telephone")
    try:
        if cell_ratio is not None and tel_ratio is not None:
            if float(cell_ratio) >= float(tel_ratio):
                channel = "mobile push"
            else:
                channel = "phone"
    except Exception:
        pass

    # month (varsa)
    month = str(prof_row.get("month", "")).lower()
    recent_months = ["mar", "apr", "may", "jun", "jul", "aug"]
    if month in recent_months:
        channel = "mobile push"
    elif month not in ["", "nan", "none"]:
        channel = "phone"

    # Zorunlu çeşitlilik: eğer hâlâ mobile push ise yaşa göre güncelle
    if channel == "mobile push" and age is not None:
        if 40 <= age < 55:
            channel = "email"
        elif age >= 55:
            channel = "phone/branch"

    return channel


# -------------------------
# Teklif seçimi
# -------------------------

def _offers_for_profile(prof_row: pd.Series, risk: str) -> str:
    offers = []

    # Bankacılığa özel temel teklifler
    offers += ["auto-bill bonus", "savings rate boost", "installment/card limit bundle"]

    # Bakiye sinyali
    balance = float(prof_row.get("balance", 0) or 0)
    if balance >= 1500:
        offers.append("wealth management check-up")

    # Konut/kredi sinyali
    has_housing = (prof_row.get("housing=yes") or 0) > 0.40
    has_loan = (prof_row.get("loan=yes") or 0) > 0.25
    if has_housing or has_loan:
        offers.append("refinance / lower-rate loan offer")

    # Kampanya yoğunluğu
    campaign = float(prof_row.get("campaign", 0) or 0)
    if campaign and campaign >= 3:
        offers.append("reduced-contact cadence")

    # Risk seviyesine göre farklı uzunlukta öneriler
    if risk == "high":
        offers += ["loyalty program + fee waiver", "interest-rate discount"]
        max_offers = 5
    elif risk == "medium":
        offers += ["personalized advisory session", "targeted card perks"]
        max_offers = 3
    else:  # low
        offers += ["informational newsletter", "light cross-sell"]
        max_offers = 2

    # Yinelenenleri kaldır ve kısalt
    dedup = []
    for o in offers:
        if o not in dedup:
            dedup.append(o)
    return ", ".join(dedup[:max_offers])


# -------------------------
# Snapshot
# -------------------------

def _snapshot_text(age: Optional[float], bal: Optional[float],
                   real: Optional[float], pred: Optional[float]) -> str:
    parts = []
    if age is not None:
        parts.append(f"Average Age: {age:.0f}")
    if bal is not None:
        parts.append(f"Balance: €{bal:.0f}")
    if real is not None:
        parts.append(f"Real Churn: {real:.0f}%")
    if pred is not None:
        parts.append(f"Predicted: {pred:.0f}%")
    return " | ".join(parts)


# -------------------------
# Ana API
# -------------------------

def recommend_for_clusters(profile_tbl: pd.DataFrame,
                           comparison_tbl: pd.DataFrame) -> pd.DataFrame:
    prof = profile_tbl.copy()
    comp = comparison_tbl.copy()
    if prof.index.name != "Cluster":
        prof.index.name = "Cluster"
    if comp.index.name != "Cluster":
        if "Cluster" in comp.columns:
            comp = comp.set_index("Cluster")
        else:
            comp.columns = ["Cluster"] + list(comp.columns[1:])
            comp = comp.set_index("Cluster")

    stats = _compute_data_thresholds(comp)

    out_rows = []
    all_clusters = sorted(set(prof.index).union(set(comp.index)))

    for cid in all_clusters:
        prof_row = prof.loc[cid] if cid in prof.index else pd.Series(dtype="float64")
        comp_row = comp.loc[cid] if cid in comp.index else pd.Series(dtype="float64")

        real, pred, mx = _extract_scores(comp_row)
        risk = _risk_level_data(mx, stats=stats)

        channel = _primary_channel(prof_row)
        offers = _offers_for_profile(prof_row, risk)

        age_val = prof_row.get("age", None)
        bal_val = prof_row.get("balance", None)
        age = float(age_val) if age_val is not None else None
        bal = float(bal_val) if bal_val is not None else None

        snapshot = _snapshot_text(age, bal, real if real else None, pred if pred else None)

        out_rows.append({
            "Cluster": cid,
            "Risk": risk,
            "Primary_Channel": channel,
            "Recommended_Offers": offers,
            "Profile_Snapshot": snapshot
        })

    return pd.DataFrame(out_rows).sort_values("Cluster").reset_index(drop=True)