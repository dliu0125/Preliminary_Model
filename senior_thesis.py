# imports
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# load and clean TRACE data
print(f"Loading {FILE_NAME} ...")
df = pd.read_csv(FILE_NAME)

# filtering through trd_exctn_dt, company_symbol, yld_pt 
required_cols = {"trd_exctn_dt", "company_symbol", "yld_pt"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# parse date and convert yld_pt to number format
df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"])
df["yld_pt"] = pd.to_numeric(df["yld_pt"], errors="coerce")

# remove rows with either missing yield or company symbol 
df = df.dropna(subset=["company_symbol", "yld_pt"])

print("Number of trades after cleaning:", len(df))

# median yield
daily_yields = (
    df.groupby(["trd_exctn_dt", "company_symbol"])["yld_pt"]
      .median()
      .reset_index()
)

panel = daily_yields.pivot(index="trd_exctn_dt",
                           columns="company_symbol",
                           values="yld_pt").sort_index()

print("Panel shape (dates x companies):", panel.shape)
print("Companies found:", list(panel.columns))

# filter to companies with enough observations (in this case 10)
min_days_per_company = 10
valid_cols = [c for c in panel.columns if panel[c].count() >= min_days_per_company]
panel = panel[valid_cols]
print("Companies kept after min-days filter:", len(valid_cols))

# running Johansen test, returning (is_cointegrated, beta_vector, trace_stat, crit_value)
def johansen_cointegration_test(series1, series2, det_order=0, k_ar_diff=1, significance=0.95):
    data = np.column_stack([series1, series2])
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < 10:
        return False, None, np.nan, np.nan

    jres = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    # rank >= 1 if trace stat for r=0 is greater than critical value
    trace_stat = jres.lr1[0]           
    crit_95 = jres.cvt[0, 1]        

    is_coint = trace_stat > crit_95
    beta = jres.evec[:, 0]
    beta = beta / beta[0]

    return is_coint, beta, trace_stat, crit_95

# mean-reversion, returning (phi, half_life, t_stat, n_obs) using e_t = a + phi * e_{t-1} + eps_t
def estimate_mean_reversion(residual):
    e = pd.Series(residual).dropna()
    if len(e) < 10:
        return np.nan, np.nan, np.nan, len(e)

    e_lag = e.shift(1).dropna()
    e_now = e.loc[e_lag.index]

    X = sm.add_constant(e_lag.values)
    y = e_now.values
    model = sm.OLS(y, X).fit()

    a = model.params[0]
    phi = model.params[1]
    t_stat = model.tvalues[1]
    n_obs = int(model.nobs)

    if abs(phi) < 1:
        half_life = -np.log(2) / np.log(abs(phi))
    else:
        half_life = np.inf

    return phi, half_life, t_stat, n_obs

# going over all company pairings to see for relationships with a minimum of 15 overlapping days
min_overlap_days = 15 

results = []

companies = list(panel.columns)
print(f"Running pairwise tests over {len(companies)} companies "
      f"({len(companies)*(len(companies)-1)//2} pairs).")

for c1, c2 in itertools.combinations(companies, 2):
    pair_df = panel[[c1, c2]].dropna()
    n_overlap = len(pair_df)
    if n_overlap < min_overlap_days:
        continue
        
    is_coint, beta, trace_stat, crit_95 = johansen_cointegration_test(
        pair_df[c1].values,
        pair_df[c2].values
    )

    if not is_coint:
        results.append({
            "issuer_1": c1,
            "issuer_2": c2,
            "n_overlap_days": n_overlap,
            "cointegrated": False,
            "trace_stat": trace_stat,
            "crit_95": crit_95,
            "phi": np.nan,
            "half_life_days": np.nan,
            "phi_t_stat": np.nan
        })
        continue

    # if pairs are cointegrated, build residual and estimate AR(1)
    S = pair_df[[c1, c2]].values
    residual = S @ beta  # beta' * [y1, y2]

    phi, half_life, phi_t, n_obs = estimate_mean_reversion(residual)

    results.append({
        "issuer_1": c1,
        "issuer_2": c2,
        "n_overlap_days": n_overlap,
        "cointegrated": True,
        "trace_stat": trace_stat,
        "crit_95": crit_95,
        "phi": phi,
        "half_life_days": half_life,
        "phi_t_stat": phi_t
    })

# results
results_df = pd.DataFrame(results)

print("\nAll pairwise results (including non-cointegrated pairs):")
display(results_df)

# |phi| < 1 filtering
cointegrated_df = results_df[
    (results_df["cointegrated"]) &
    (results_df["phi"].abs() < 1)
].copy()

# sort from short to long half-life
cointegrated_df = cointegrated_df.sort_values("half_life_days")

print("\nCointegrated pairs with mean-reverting residuals (sorted by half-life):")
display(cointegrated_df)

# save results to CSV
results_df.to_csv("pairwise_johansen_results_all_pairs.csv", index=False)
cointegrated_df.to_csv("pairwise_johansen_results_cointegrated.csv", index=False)

print("\nSaved:")
print(" - pairwise_johansen_results_all_pairs.csv")
print(" - pairwise_johansen_results_cointegrated.csv")
