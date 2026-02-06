# app.py
# Streamlit end-to-end: read CSV from a fixed path -> clean/profile -> EDA -> models -> partial plot -> PDF download
# FIXES APPLIED:
# 1) Removed ALL file-uploader logic (no `uploaded`, no `.seek`)
# 2) Uses your fixed CSV_PATH everywhere
# 3) Keeps OUT_DIR creation (per your request) but DOES NOT rely on saving plots to disk
# 4) Fixes Arrow serialization warnings by sanitizing dtypes before st.dataframe
# 5) Ensures all helper functions exist (safe_numeric, build_controls, etc.)

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import statsmodels.api as sm
import statsmodels.formula.api as smf

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


# -----------------------------
# Config (your requested pattern)
# -----------------------------
CSV_PATH = "testdata.csv"
OUT_DIR = "top_revenue_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Helper functions
# -----------------------------
def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric; invalid parses become NaN."""
    return pd.to_numeric(series, errors="coerce")


def sanitize_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit can warn about Arrow serialization with mixed dtypes / extension dtypes.
    Make a display-safe copy (keeps analysis df unchanged).
    """
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
        elif pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def build_controls(df: pd.DataFrame, y_col: str, x_col: str):
    """Return (controls list, terms list) where object controls become C(col)."""
    controls = [c for c in df.columns if c not in [y_col, x_col]]
    terms = []
    for c in controls:
        if pd.api.types.is_object_dtype(df[c]):
            terms.append(f"C({c})")
        else:
            terms.append(c)
    return controls, terms


def corr_stats(df: pd.DataFrame, x: str, y: str):
    pearson = df[[x, y]].corr(method="pearson").iloc[0, 1]
    spearman = df[[x, y]].corr(method="spearman").iloc[0, 1]
    return float(pearson), float(spearman)


def model_with_robust_se(formula: str, df: pd.DataFrame):
    """OLS with robust (HC3) SE."""
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def coef_ci_robust(model, term: str = "top"):
    """(coef, CI_low, CI_high) using model robust SE."""
    b = float(model.params.get(term, np.nan))
    se = float(model.bse.get(term, np.nan))
    return b, b - 1.96 * se, b + 1.96 * se


def fig_scatter_lowess(df: pd.DataFrame, x="top", y="revenue", frac=0.25):
    fig = plt.figure(figsize=(8, 4.8))
    plt.scatter(df[x], df[y], s=10, alpha=0.25)
    lowess = sm.nonparametric.lowess(df[y], df[x], frac=frac, return_sorted=True)
    plt.plot(lowess[:, 0], lowess[:, 1], linewidth=2)
    plt.xlabel("Time on page (top)")
    plt.ylabel("Revenue")
    plt.title("Revenue vs Time on Page (scatter + smooth trend)")
    plt.tight_layout()
    return fig


def fig_binned_means(df: pd.DataFrame, x="top", y="revenue", q=20):
    bins = pd.qcut(df[x], q=q, duplicates="drop")
    g = df.groupby(bins, observed=True).agg(
        x_mean=(x, "mean"),
        y_mean=(y, "mean"),
        y_std=(y, "std"),
        n=(y, "size"),
    ).reset_index(drop=True)

    g["y_se"] = g["y_std"] / np.sqrt(g["n"])
    g["ci_lo"] = g["y_mean"] - 1.96 * g["y_se"]
    g["ci_hi"] = g["y_mean"] + 1.96 * g["y_se"]

    X = g["x_mean"].astype(float).to_numpy()
    Y = g["y_mean"].astype(float).to_numpy()
    LO = g["ci_lo"].astype(float).to_numpy()
    HI = g["ci_hi"].astype(float).to_numpy()

    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(X, Y, marker="o")
    plt.fill_between(X, LO, HI, alpha=0.2)
    plt.xlabel("Time on page (top) — bin means")
    plt.ylabel("Mean revenue (95% CI)")
    plt.title("Average revenue across time-on-page bins")
    plt.tight_layout()
    return fig


def fig_partial_relationship(df: pd.DataFrame, terms, x="top", y="revenue", frac=0.25):
    """
    Residualize y on controls and x on controls (FWL), then plot residual-residual relationship.
    """
    if terms:
        m_y = smf.ols(f"{y} ~ " + " + ".join(terms), data=df).fit()
        y_res = m_y.resid
        m_x = smf.ols(f"{x} ~ " + " + ".join(terms), data=df).fit()
        x_res = m_x.resid
    else:
        y_res = df[y] - df[y].mean()
        x_res = df[x] - df[x].mean()

    fig = plt.figure(figsize=(8, 4.8))
    plt.scatter(x_res, y_res, s=10, alpha=0.25)
    lw = sm.nonparametric.lowess(y_res, x_res, frac=frac, return_sorted=True)
    plt.plot(lw[:, 0], lw[:, 1], linewidth=2)
    plt.xlabel("top (variation after controls)")
    plt.ylabel("revenue (variation after controls)")
    plt.title("Relationship after controlling for other variables")
    plt.tight_layout()
    return fig


def render_pdf_bytes(
    pearson_r, spearman_rho,
    effect10_simple, ci10_simple,
    effect10_ctrl, ci10_ctrl,
    r2_simple, r2_ctrl,
    fig1, fig2, fig3
) -> bytes:
    """Generate a short readable PDF report in-memory (no saving to disk)."""

    def fig_to_png_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        return buf

    img1 = fig_to_png_bytes(fig1)
    img2 = fig_to_png_bytes(fig2)
    img3 = fig_to_png_bytes(fig3)

    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buf,
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.grey))

    def bullets(items):
        return "<br/>".join([f"• {x}" for x in items])

    summary = [
        f"Time on page (top) and revenue move together (Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_rho:.3f}).",
        f"Simple model: +10 units of time on page is associated with about {effect10_simple:.5f} more revenue "
        f"(95% CI {ci10_simple[0]:.5f} to {ci10_simple[1]:.5f}).",
        f"Controlled model: after accounting for other variables, the association is about {effect10_ctrl:.5f} per +10 units "
        f"(95% CI {ci10_ctrl[0]:.5f} to {ci10_ctrl[1]:.5f}).",
        f"Model fit improves with controls (R² {r2_simple:.3f} → {r2_ctrl:.3f}).",
    ]

    metrics_table = [
        ["Model", "Effect of +10 top units", "95% CI", "R²"],
        ["Simple: revenue ~ top", f"{effect10_simple:.5f}", f"[{ci10_simple[0]:.5f}, {ci10_simple[1]:.5f}]", f"{r2_simple:.3f}"],
        ["Controlled: + other variables", f"{effect10_ctrl:.5f}", f"[{ci10_ctrl[0]:.5f}, {ci10_ctrl[1]:.5f}]", f"{r2_ctrl:.3f}"],
    ]

    story = []
    story.append(Paragraph("Time on Page (top) vs Revenue — Brief Report", styles["TitleX"]))
    story.append(Paragraph(
        "This report is intended for a mixed technical and non-technical audience. "
        "It summarizes the relationship between time on page and revenue using clear visuals and regression models.",
        styles["BodyX"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Executive summary", styles["H2X"]))
    story.append(Paragraph(bullets(summary), styles["BodyX"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Visual relationship", styles["H2X"]))
    story.append(RLImage(img1, width=6.6 * inch, height=3.7 * inch))
    story.append(Spacer(1, 6))
    story.append(RLImage(img2, width=6.6 * inch, height=3.7 * inch))
    story.append(Spacer(1, 10))

    story.append(Paragraph("After controlling for other variables", styles["H2X"]))
    story.append(RLImage(img3, width=6.6 * inch, height=3.7 * inch))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Model summary (readable)", styles["H2X"]))
    tbl = Table(metrics_table, colWidths=[2.4 * inch, 2.2 * inch, 1.4 * inch, 0.6 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("FONTSIZE", (0, 1), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Interpretation note: these are associations, not necessarily causal effects. "
        "Controlling for other variables helps test whether the observed relationship is driven by differences across groups.",
        styles["SmallX"]
    ))

    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf.read()


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Top vs Revenue Analysis", layout="wide")
st.title("Relationship Between Time on Page (top) and Revenue")

st.caption(f"Reading CSV from: {CSV_PATH}")

# Read CSV (no uploader)
try:
    df_raw = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to read CSV at {CSV_PATH}: {e}")
    st.stop()

st.subheader("1) Data cleaning + searching (profiling)")
st.write("Checks missing values, data types, duplicates, and ensures `top` and `revenue` are numeric.")

# Profiling
profile = pd.DataFrame({
    "column": df_raw.columns,
    "dtype": [str(df_raw[c].dtype) for c in df_raw.columns],
    "missing_n": [int(df_raw[c].isna().sum()) for c in df_raw.columns],
    "missing_pct": [float(df_raw[c].isna().mean()) for c in df_raw.columns],
    "nunique": [int(df_raw[c].nunique(dropna=True)) for c in df_raw.columns],
}).sort_values("missing_pct", ascending=False)

col1, col2 = st.columns([1, 1])
with col1:
    st.write("Preview")
    st.dataframe(sanitize_for_streamlit(df_raw.head(10)), use_container_width=True)
with col2:
    st.write("Column profile")
    st.dataframe(sanitize_for_streamlit(profile), use_container_width=True)

dup_n = int(df_raw.duplicated().sum())
st.write(f"Duplicate rows: **{dup_n:,}**")

required_cols = {"top", "revenue"}
missing_required = required_cols - set(df_raw.columns)
if missing_required:
    st.error(f"Your file is missing required columns: {missing_required}. It must contain 'top' and 'revenue'.")
    st.stop()

# Clean
df = df_raw.copy()
df["top"] = safe_numeric(df["top"])
df["revenue"] = safe_numeric(df["revenue"])

before_n = len(df)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["top", "revenue"]).reset_index(drop=True)
df = df[(df["top"] >= 0) & (df["revenue"] >= 0)].reset_index(drop=True)
after_n = len(df)

st.write(f"Rows before cleaning: **{before_n:,}**")
st.write(f"Rows after cleaning: **{after_n:,}**")

# Identify controls + terms
controls, terms = build_controls(df, y_col="revenue", x_col="top")

# Stabilize categoricals for modeling
for c in controls:
    if pd.api.types.is_object_dtype(df[c]):
        df[c] = df[c].astype(str)

st.write("Detected control variables (all other columns besides `top` and `revenue`):")
st.code(", ".join(controls) if controls else "None", language="text")

st.subheader("2) EDA (visual + statistical)")
st.write("We visualize the raw relationship and compute correlation as a quick signal check.")

pearson_r, spearman_rho = corr_stats(df, "top", "revenue")

mcol1, mcol2 = st.columns(2)
with mcol1:
    st.metric("Pearson correlation (linear)", f"{pearson_r:.3f}")
with mcol2:
    st.metric("Spearman rho (rank-based)", f"{spearman_rho:.3f}")

fig1 = fig_scatter_lowess(df, x="top", y="revenue")
st.pyplot(fig1)

fig2 = fig_binned_means(df, x="top", y="revenue", q=20)
st.pyplot(fig2)

st.subheader("3) Modeling (simple vs controlled)")
st.write(
    "We fit two models:\n"
    "- **Simple:** `revenue ~ top`\n"
    "- **Controlled:** `revenue ~ top + (all other variables)`\n\n"
    "Then we compare how the `top` effect changes."
)

# Model A
m1 = model_with_robust_se("revenue ~ top", df)

# Model B
formula_m2 = "revenue ~ top"
if terms:
    formula_m2 += " + " + " + ".join(terms)
m2 = model_with_robust_se(formula_m2, df)

b1, lo1, hi1 = coef_ci_robust(m1, "top")
b2, lo2, hi2 = coef_ci_robust(m2, "top")

effect10_simple = 10 * b1
effect10_ctrl = 10 * b2
ci10_simple = (10 * lo1, 10 * hi1)
ci10_ctrl = (10 * lo2, 10 * hi2)

t1, t2 = st.columns(2)
with t1:
    st.markdown("### Simple model")
    st.write("Association between time on page and revenue without controlling for other variables.")
    st.metric("Effect of +10 top units on revenue", f"{effect10_simple:.5f}")
    st.write(f"95% CI: [{ci10_simple[0]:.5f}, {ci10_simple[1]:.5f}]")
    st.write(f"R²: {m1.rsquared:.3f}")

with t2:
    st.markdown("### Controlled model")
    st.write("Association after controlling for other variables (categoricals handled automatically).")
    st.metric("Effect of +10 top units on revenue", f"{effect10_ctrl:.5f}")
    st.write(f"95% CI: [{ci10_ctrl[0]:.5f}, {ci10_ctrl[1]:.5f}]")
    st.write(f"R²: {m2.rsquared:.3f}")

st.write("**Does the relationship change after controlling?**")
delta = effect10_ctrl - effect10_simple
st.write(
    f"- Change in +10-unit effect: **{delta:.5f}** "
    f"(controlled minus simple). "
    "If this shifts a lot, it suggests the raw relationship was partly driven by differences across other variables."
)

st.write("Below is the **partial relationship** after removing the influence of control variables (residualized view).")
fig3 = fig_partial_relationship(df, terms, x="top", y="revenue")
st.pyplot(fig3)

with st.expander("Optional: show model formulas and detailed stats output (technical)"):
    st.code("Model A: revenue ~ top", language="text")
    st.code(f"Model B: {formula_m2}", language="text")
    st.write("Model A summary (technical):")
    st.text(m1.summary().as_text())
    st.write("Model B summary (technical):")
    st.text(m2.summary().as_text())

st.subheader("4) Download a brief PDF report")
st.write("This PDF is meant to be readable to mixed audiences and avoids dumping raw statistical tables.")

pdf_bytes = render_pdf_bytes(
    pearson_r=pearson_r,
    spearman_rho=spearman_rho,
    effect10_simple=effect10_simple,
    ci10_simple=ci10_simple,
    effect10_ctrl=effect10_ctrl,
    ci10_ctrl=ci10_ctrl,
    r2_simple=float(m1.rsquared),
    r2_ctrl=float(m2.rsquared),
    fig1=fig1, fig2=fig2, fig3=fig3
)

st.download_button(
    label="Download PDF report",
    data=pdf_bytes,
    file_name="top_vs_revenue_report.pdf",
    mime="application/pdf"
)