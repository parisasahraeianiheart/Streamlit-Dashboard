# -----------------------------
# Config
# -----------------------------
CSV_PATH = "testdata.csv"
OUT_DIR = "top_revenue_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load
# -----------------------------
df_raw = pd.read_csv(CSV_PATH)
print("Shape:", df_raw.shape)
print("Columns:", df_raw.columns.tolist())
display(df_raw.head(5))

# -----------------------------
# Quick "searching"/profiling
# -----------------------------
profile = pd.DataFrame({
    "column": df_raw.columns,
    "dtype": [df_raw[c].dtype for c in df_raw.columns],
    "missing_n": [df_raw[c].isna().sum() for c in df_raw.columns],
    "missing_pct": [df_raw[c].isna().mean() for c in df_raw.columns],
    "nunique": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
})
display(profile.sort_values("missing_pct", ascending=False))

# Duplicates
dup_n = df_raw.duplicated().sum()
print("Duplicate rows:", dup_n)

# -----------------------------
# Standard cleaning assumptions
# -----------------------------
# 1) Ensure 'top' and 'revenue' are numeric
# 2) Drop rows missing either key field (top, revenue)
# 3) Optional: clip extreme outliers for plots (not for model unless you choose)
# -----------------------------

df = df_raw.copy()

# coerce numeric
for col in ["top", "revenue"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# drop missing key vars
df = df.dropna(subset=["top", "revenue"]).reset_index(drop=True)

# remove negative values if they don't make sense in your context
# (keep this conservative; adjust to your domain rules)
df = df[(df["top"] >= 0) & (df["revenue"] >= 0)].reset_index(drop=True)

print("After cleaning:", df.shape)

# -----------------------------
# Identify candidate control variables
# -----------------------------
# We'll control for "other columns" besides top and revenue.
control_cols = [c for c in df.columns if c not in ["top", "revenue"]]
print("Control columns:", control_cols)

# Treat low-cardinality strings/ids as categoricals
# (If you have continuous controls, they'll remain numeric.)
for c in control_cols:
    # If it's object/string, set to str so statsmodels can do C(var)
    if df[c].dtype == "object":
        df[c] = df[c].astype(str)

# Basic numeric sanity summary
display(df[["top", "revenue"]].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T)

# -----------------------------
# Basic correlation (quick signal check)
# -----------------------------
pearson_r = df[["top", "revenue"]].corr(method="pearson").iloc[0, 1]
spearman_rho = df[["top", "revenue"]].corr(method="spearman").iloc[0, 1]
print(f"Pearson r: {pearson_r:.4f}")
print(f"Spearman rho: {spearman_rho:.4f}")

# -----------------------------
# Plot 1: scatter + LOWESS smooth
# -----------------------------
fig = plt.figure(figsize=(8, 4.8))
plt.scatter(df["top"], df["revenue"], s=10, alpha=0.25)
lowess = sm.nonparametric.lowess(df["revenue"], df["top"], frac=0.25, return_sorted=True)
plt.plot(lowess[:, 0], lowess[:, 1], linewidth=2)

plt.xlabel("Time on page (top)")
plt.ylabel("Revenue")
plt.title("Revenue vs Time on Page (scatter + smooth trend)")
p_scatter = os.path.join(OUT_DIR, "scatter_lowess.png")
plt.tight_layout()
plt.savefig(p_scatter, dpi=200)
plt.show()

# -----------------------------
# Plot 2: binned averages with CI (reduces noise)
# -----------------------------
bins = pd.qcut(df["top"], q=20, duplicates="drop")
g = df.groupby(bins, observed=True).agg(
    top_mean=("top", "mean"),
    rev_mean=("revenue", "mean"),
    rev_std=("revenue", "std"),
    n=("revenue", "size")
).reset_index(drop=True)

g["rev_se"] = g["rev_std"] / np.sqrt(g["n"])
g["ci_lo"] = g["rev_mean"] - 1.96 * g["rev_se"]
g["ci_hi"] = g["rev_mean"] + 1.96 * g["rev_se"]

x = g["top_mean"].astype(float).to_numpy()
y = g["rev_mean"].astype(float).to_numpy()
lo = g["ci_lo"].astype(float).to_numpy()
hi = g["ci_hi"].astype(float).to_numpy()

fig = plt.figure(figsize=(8, 4.8))
plt.plot(x, y, marker="o")
plt.fill_between(x, lo, hi, alpha=0.2)
plt.xlabel("Time on page (top) — bin means")
plt.ylabel("Mean revenue (95% CI)")
plt.title("Average revenue across time-on-page bins")
p_bins = os.path.join(OUT_DIR, "binned_means.png")
plt.tight_layout()
plt.savefig(p_bins, dpi=200)
plt.show()

# -----------------------------
# Optional: Group comparison (helps a nontechnical audience)
# e.g. does the pattern differ by platform/browser/site?
# -----------------------------
for group_col in ["platform", "browser", "site"]:
    if group_col in df.columns:
        tmp = df.groupby(group_col, observed=True).agg(
            n=("revenue", "size"),
            top_avg=("top", "mean"),
            rev_avg=("revenue", "mean")
        ).sort_values("n", ascending=False).head(10)
        print(f"\nTop 10 groups by count for {group_col}:")
        display(tmp)


# -----------------------------
# Build formulas
# -----------------------------
controls = [c for c in df.columns if c not in ["top", "revenue"]]

# Model A: simple
m1 = smf.ols("revenue ~ top", data=df).fit(cov_type="HC3")  # robust SE helps with heteroskedasticity

# Model B: controlled
# Treat object columns as categorical via C(col). Numeric controls go in directly.
terms = []
for c in controls:
    if df[c].dtype == "object":
        terms.append(f"C({c})")
    else:
        terms.append(c)

formula_m2 = "revenue ~ top"
if terms:
    formula_m2 += " + " + " + ".join(terms)

m2 = smf.ols(formula_m2, data=df).fit(cov_type="HC3")

# -----------------------------
# Extract interpretable results (no raw output)
# -----------------------------
def coef_ci(model, term="top"):
    b = float(model.params.get(term, np.nan))
    se = float(model.bse.get(term, np.nan))
    return b, b - 1.96*se, b + 1.96*se

b1, lo1, hi1 = coef_ci(m1, "top")
b2, lo2, hi2 = coef_ci(m2, "top")

print("Model A (simple): top coef =", b1, "CI:", (lo1, hi1), "R2:", m1.rsquared)
print("Model B (controlled): top coef =", b2, "CI:", (lo2, hi2), "R2:", m2.rsquared)

# Optional: interpret per +10 units of time-on-page
print("\nEffect per +10 top units:")
print("Simple:", 10*b1, " (CI:", (10*lo1, 10*hi1), ")")
print("Controlled:", 10*b2, " (CI:", (10*lo2, 10*hi2), ")")

# -----------------------------
# Partial relationship plot (Frisch–Waugh–Lovell style):
# residualize revenue and top on controls, then plot residuals
# -----------------------------
if terms:
    # revenue residuals after controls
    m_y = smf.ols("revenue ~ " + " + ".join(terms), data=df).fit()
    y_res = m_y.resid

    # top residuals after controls
    m_x = smf.ols("top ~ " + " + ".join(terms), data=df).fit()
    x_res = m_x.resid
else:
    y_res = df["revenue"] - df["revenue"].mean()
    x_res = df["top"] - df["top"].mean()

fig = plt.figure(figsize=(8, 4.8))
plt.scatter(x_res, y_res, s=10, alpha=0.25)
lw = sm.nonparametric.lowess(y_res, x_res, frac=0.25, return_sorted=True)
plt.plot(lw[:, 0], lw[:, 1], linewidth=2)

plt.xlabel("top (variation after controls)")
plt.ylabel("revenue (variation after controls)")
plt.title("Relationship after controlling for other variables")
p_partial = os.path.join(OUT_DIR, "partial_relationship.png")
plt.tight_layout()
plt.savefig(p_partial, dpi=200)
plt.show()