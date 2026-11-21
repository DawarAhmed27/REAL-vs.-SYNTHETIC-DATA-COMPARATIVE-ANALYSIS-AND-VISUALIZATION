"""Phase 6: comparison of the three scenarios (Real, Synthetic, Combined).

Everything here takes a dictionary ``dfs = {"Real": ..., "Synthetic": ..., "Combined": ...}`` and uses the
shared settings of ``scenario_lib`` (same x-limits, bins and y-limits as in notebooks 04 to 06), so the
comparison is fair. The Python charts in this module are the ones Power BI cannot draw (correlation
matrices) plus Python versions of the Power BI comparison charts, which are rebuilt natively in Power BI.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import scenario_lib as L

SC = ["Real", "Synthetic", "Combined"]
COL = L.SCENARIO_COLOR
DIST_COLS = ["PageValues", "ProductRelated_Duration", "Total_Pages", "ExitRates", "Avg_Time_Per_Page", "Administrative"]
BOX_COLS = ["ProductRelated_Duration", "Total_Pages", "Total_Duration", "Avg_Time_Per_Page", "ExitRates", "PageValues"]
CAT_COLS = ["Month", "Quarter", "VisitorType", "Weekend", "Session_Length_Band", "Revenue"]
REL_PAIRS = [("ProductRelated", "ProductRelated_Duration"), ("BounceRates", "ExitRates")]


def make_scenarios(real, syn):
    comb = L.combine(real, syn)
    return {"Real": real, "Synthetic": syn, "Combined": comb.drop(columns="Data_Source")}


# ---------------------------------------------------------------------------------------------- tables
def stats_table(dfs, cols=L.NUM):
    """Mean, median, standard deviation, minimum and maximum of each column in each scenario (long format)."""
    rows = []
    for c in cols:
        for s in SC:
            x = dfs[s][c]
            rows.append({"column": c, "scenario": s, "mean": x.mean(), "median": x.median(), "std": x.std(),
                         "min": x.min(), "max": x.max()})
    return pd.DataFrame(rows)


def distance_to_real(dfs):
    """Per numerical column: KS distance, standardised mean difference and std ratio of Synthetic and Combined to Real."""
    r = dfs["Real"]; rows = []
    for c in L.NUM:
        for s in ("Synthetic", "Combined"):
            x = dfs[s][c]; sd = np.sqrt((r[c].var() + x.var()) / 2)
            rows.append({"column": c, "scenario": s, "KS": stats.ks_2samp(r[c], x).statistic,
                         "mean diff (%)": 100 * (x.mean() - r[c].mean()) / r[c].mean() if r[c].mean() else np.nan,
                         "Cohen d": (x.mean() - r[c].mean()) / sd if sd else np.nan,
                         "std ratio": x.std() / r[c].std() if r[c].std() else np.nan})
    return pd.DataFrame(rows)


def correlation_long(dfs):
    out = []
    for s in SC:
        p = dfs[s][L.NUM].corr(); sp = dfs[s][L.NUM].corr(method="spearman")
        for i, a in enumerate(L.NUM):
            for b in L.NUM[i + 1:]:
                out.append({"scenario": s, "variable_1": a, "variable_2": b, "pearson": p.loc[a, b], "spearman": sp.loc[a, b]})
    return pd.DataFrame(out)


def categorical_shares(dfs):
    out = []
    for s in SC:
        for c in L.CAT:
            vc = dfs[s][c].value_counts(normalize=True) * 100
            for lev, v in vc.items():
                out.append({"scenario": s, "column": c, "level": lev, "share_pct": v})
    return pd.DataFrame(out)


def _v_plain(a, b):
    """Cramer's V from the chi-square statistic WITHOUT the continuity correction (matters only for 2 x 2 tables).
    Returns (chi2, p, V)."""
    ct = pd.crosstab(a, b); chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    k = min(ct.shape) - 1
    return chi2, p, (np.sqrt(chi2 / (ct.values.sum() * k)) if k > 0 else np.nan)


def categorical_match(dfs):
    """Cramer's V between data source and each categorical column (Real vs Synthetic) and the largest share gap."""
    r, s = dfs["Real"], dfs["Synthetic"]; rows = []
    for c in L.CAT:
        both = pd.concat([r[c], s[c]], ignore_index=True); src = np.r_[np.zeros(len(r)), np.ones(len(s))]
        chi2, p, v = _v_plain(both, src)
        sh = pd.concat([r[c].value_counts(normalize=True), s[c].value_counts(normalize=True)], axis=1).fillna(0) * 100
        rows.append({"column": c, "levels": sh.shape[0], "chi2": chi2, "p-value": p, "Cramer's V": v, "largest share gap (pp)": (sh.iloc[:, 1] - sh.iloc[:, 0]).abs().max()})
    return pd.DataFrame(rows).set_index("column")


def monthly_long(dfs):
    out = []
    for s in SC:
        t = L.monthly_table(dfs[s]).reset_index(); t.insert(0, "scenario", s); out.append(t)
    return pd.concat(out, ignore_index=True)


def driver_effects(df):
    """Effect size of each attribute on purchase: rank-biserial (numerical) and Cramer's V (categorical)."""
    eff = {}
    for v in ["PageValues", "ExitRates", "Total_Pages", "Total_Duration", "BounceRates", "Avg_Time_Per_Page"]:
        eff[f"{v} (rank-biserial)"] = abs(L._rank_biserial(df, v))
    for c in ["Session_Length_Band", "Month", "Quarter", "VisitorType", "Weekend"]:
        eff[f"{c} (Cramer's V)"] = _v_plain(df[c], df.Revenue)[2]
    return pd.Series(eff)


# ---------------------------------------------------------------------------------------------- figures
def fig_three_way_distribution(dfs, S, path, cols=DIST_COLS):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2)); axes = axes.ravel()
    for ax, c in zip(axes, cols):
        for s in SC:
            h = L._hist_pct(dfs[s][c].clip(upper=S["hi"][c]), S["bins"][c]); e = S["bins"][c]
            ax.stairs(h, e, color=COL[s], lw=1.6, label=s)
        ax.set_xlim(e[0], e[-1]); ax.set_ylim(0, S["hist_ymax"][c]); ax.set_title(c); ax.set_ylabel("% of rows")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Three-way distribution: Real, Synthetic and Combined on identical bins (values above the real 99th percentile fall in the last bin)", y=1.0, fontsize=11)
    fig.tight_layout(); L.save(fig, path)


def fig_mean_median_std(tab, path):
    """Mean, median and standard deviation of every column, as a percentage of the Real value (Real = 100)."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5)); w = 0.27
    for ax, st in zip(axes, ["mean", "median", "std"]):
        wide = tab.pivot(index="column", columns="scenario", values=st).loc[L.NUM, SC]
        ok = wide["Real"] > 0; idx = wide[ok]
        for k, s in enumerate(SC):
            ax.bar(np.arange(len(idx)) + (k - 1) * w, 100 * idx[s] / idx["Real"], w, color=COL[s], label=s)
        ax.axhline(100, color="k", lw=.8); ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("% of the Real value")
        note = "" if ok.all() else "   (not shown, real value is 0: " + ", ".join(wide.index[~ok]) + ")"
        ax.set_title(f"{st.capitalize() if st != 'std' else 'Standard deviation'} relative to Real{note}", fontsize=10)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout(); L.save(fig, path)


def fig_boxes(dfs, S, path, cols=BOX_COLS):
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.6))
    for ax, c in zip(axes, cols):
        data = [dfs[s][c].values for s in SC]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=.6, medianprops={"color": "k"})
        for patch, s in zip(bp["boxes"], SC): patch.set_facecolor(COL[s]); patch.set_alpha(.8)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Real", "Syn.", "Comb."], fontsize=8)
        ax.set_ylim(0, S["hi"][c]); ax.set_title(c, fontsize=9)
    fig.suptitle("Comparative box plots (outlier points hidden; y-axis limit = real 99th percentile)", y=1.02, fontsize=11)
    fig.tight_layout(); L.save(fig, path)


def fig_corr_side_by_side(dfs, method, path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6))
    cbar_ax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    for i, (ax, s) in enumerate(zip(axes, SC)):
        m = dfs[s][L.NUM].corr(method=method)
        sns.heatmap(m, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 5.5}, linewidths=.3,
                    cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None, square=True)
        ax.set_title(f"{s}", fontsize=11); ax.tick_params(labelsize=7)
        if i > 0: ax.set_yticklabels([])
    fig.suptitle(f"{method.capitalize()} correlation matrices of the numerical columns, side by side", y=1.0, fontsize=12)
    fig.subplots_adjust(wspace=0.05, right=0.91); L.save(fig, path)


def fig_categorical(dfs, S, path, cols=CAT_COLS):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6)); axes = axes.ravel(); w = 0.27
    for ax, c in zip(axes, cols):
        lev = L.levels_for(dfs["Real"], c)
        for k, s in enumerate(SC):
            sh = 100 * dfs[s][c].value_counts(normalize=True).reindex(lev).fillna(0)
            ax.bar(np.arange(len(lev)) + (k - 1) * w, sh.values, w, color=COL[s], label=s)
        ax.set_xticks(range(len(lev))); ax.set_xticklabels(lev, rotation=30, ha="right", fontsize=8); ax.set_title(c); ax.set_ylabel("% of rows")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Categorical distributions in the three scenarios (share of each scenario's rows)", y=1.0, fontsize=11)
    fig.tight_layout(); L.save(fig, path)


def fig_relationship(dfs, S, path, pairs=REL_PAIRS, n=1500):
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.6 * len(pairs), 5.6))
    for ax, (x, y) in zip(np.atleast_1d(axes), pairs):
        for s in SC:
            d = dfs[s]; smp = d.sample(n, random_state=L.SEED)
            ax.scatter(smp[x].clip(upper=S["hi"][x]), smp[y].clip(upper=S["hi"][y]), s=5, alpha=.18, color=COL[s])
            slope, icpt = np.polyfit(d[x], d[y], 1); r = d[x].corr(d[y]); rho = d[x].corr(d[y], method="spearman")
            xs = np.array([0, S["hi"][x]]); ax.plot(xs, icpt + slope * xs, color=COL[s], lw=2.4,
                                                    label=f"{s}: slope {slope:.3g}, Pearson {r:.2f}, Spearman {rho:.2f}")
        ax.set_xlim(0, S["hi"][x]); ax.set_ylim(0, S["hi"][y]); ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}: least-squares line of each scenario (fitted on all rows)", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); L.save(fig, path)


def fig_trend(dfs, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for s in SC:
        t = L.monthly_table(dfs[s])
        axes[0].plot(t.index, t["rate"], marker="o", color=COL[s], label=s)
        if s != "Combined":                                    # 95% Wilson intervals (wide for months with few sessions)
            axes[0].fill_between(range(len(t)), t["lo"], t["hi"], color=COL[s], alpha=0.12, lw=0)
        axes[1].plot(t.index, t["share"], marker="o", color=COL[s], label=s)
    axes[0].set_title("Purchase rate per month (%), bands: 95% intervals"); axes[1].set_title("Share of the year's sessions per month (%)")
    axes[0].legend(frameon=False); fig.tight_layout(); L.save(fig, path)


# ---------------------------------------------------------------------------------------------- summary matrix
def summary_evidence(dfs):
    """All numbers used in the summary matrix and in the written answers, in one dictionary."""
    r, s, c = dfs["Real"], dfs["Synthetic"], dfs["Combined"]; E = {}
    dist = distance_to_real(dfs); ds = dist[dist.scenario == "Synthetic"].set_index("column"); dc = dist[dist.scenario == "Combined"].set_index("column")
    for key, d in (("syn", ds), ("comb", dc)):
        E[f"{key}_mean_diff_max_col"] = d["mean diff (%)"].abs().idxmax(); E[f"{key}_mean_diff_max"] = float(d["mean diff (%)"].loc[d["mean diff (%)"].abs().idxmax()])
        E[f"{key}_mean_within8"] = int((d["mean diff (%)"].abs() < 8).sum())
        E[f"{key}_ks_mean"] = float(d.KS.mean()); E[f"{key}_ks_gt05"] = int((d.KS > .05).sum())
        E[f"{key}_ks_max_col"] = d.KS.idxmax(); E[f"{key}_ks_max"] = float(d.KS.max())
        E[f"{key}_std_ratio_min_col"] = d["std ratio"].idxmin(); E[f"{key}_std_ratio_min"] = float(d["std ratio"].min())
        E[f"{key}_std_ratio_max_col"] = d["std ratio"].idxmax(); E[f"{key}_std_ratio_max"] = float(d["std ratio"].max())
        E[f"{key}_cohen_max"] = float(d["Cohen d"].abs().max())
    med_close = 0
    for col in L.NUM:
        a, b = r[col].median(), s[col].median()
        med_close += int((a == b == 0) or (a != 0 and abs(b - a) / abs(a) < .10))
    E["syn_median_close"] = med_close
    # medians that differ by more than 10% (list), and sensitivity of the "close" counts to the threshold
    med_far = []
    for col in L.NUM:
        a, b = r[col].median(), s[col].median()
        if not ((a == b == 0) or (a != 0 and abs(b - a) / abs(a) < .10)):
            med_far.append(f"{col} ({a:.4g} real, {b:.4g} synthetic)")
    E["syn_median_far"] = med_far
    cm_far = []
    for col in L.NUM:
        a, b = r[col].median(), c[col].median()
        if not ((a == b == 0) or (a != 0 and abs(b - a) / abs(a) < .10)):
            cm_far.append(f"{col} ({a:.4g} real, {b:.4g} combined)")
    E["comb_median_close"] = len(L.NUM) - len(cm_far); E["comb_median_far"] = cm_far
    E["thr_mean"] = {t: int((ds["mean diff (%)"].abs() < t).sum()) for t in (5, 8, 10, 15)}
    E["thr_mean_comb"] = {t: int((dc["mean diff (%)"].abs() < t).sum()) for t in (5, 8, 10, 15)}
    E["thr_ks"] = {t: int((ds.KS > t).sum()) for t in (0.03, 0.05, 0.10)}
    E["thr_ks_comb"] = {t: int((dc.KS > t).sum()) for t in (0.03, 0.05, 0.10)}
    cm = categorical_match(dfs)
    for k in ("Revenue", "Weekend"):
        E[f"chi2_{k}"] = float(cm.loc[k, "chi2"]); E[f"p_{k}"] = float(cm.loc[k, "p-value"]); E[f"V_{k}"] = float(cm.loc[k, "Cramer's V"])
    E["cat_V_min"] = float(cm["Cramer's V"].min()); E["cat_V_max"] = float(cm["Cramer's V"].max())
    E["cat_best"] = cm["Cramer's V"].idxmin(); E["cat_worst"] = cm["Cramer's V"].idxmax()
    E["cat_gap_max"] = float(cm["largest share gap (pp)"].max()); E["cat_gap_max_col"] = cm["largest share gap (pp)"].idxmax()
    E["cat_gap_best"] = float(cm.loc[E["cat_best"], "largest share gap (pp)"])
    sh_r = {k: r[k].value_counts(normalize=True) for k in L.CAT}; sh_c = {k: c[k].value_counts(normalize=True) for k in L.CAT}
    E["comb_cat_gap_max"] = float(max((100 * (sh_c[k] - sh_r[k].reindex(sh_c[k].index).fillna(0))).abs().max() for k in L.CAT))
    cl = correlation_long(dfs)
    for meth in ("pearson", "spearman"):
        w = cl.pivot_table(index=["variable_1", "variable_2"], columns="scenario", values=meth)
        for key, col in (("syn", "Synthetic"), ("comb", "Combined")):
            d = (w[col] - w["Real"]).abs(); E[f"{key}_corr_{meth}_mean"] = float(d.mean()); E[f"{key}_corr_{meth}_max"] = float(d.max())
            E[f"{key}_corr_{meth}_max_pair"] = " ~ ".join(d.idxmax())
            for thr, tag in ((.1, ""), (.3, "_r30")):
                strong = w["Real"].abs() >= thr
                E[f"{key}_corr_{meth}_signflip{tag}"] = int(((np.sign(w[col]) != np.sign(w["Real"])) & strong).sum()); E[f"corr_{meth}_strong_pairs{tag}"] = int(strong.sum())
        lo_, hi_ = np.minimum(w["Real"], w["Synthetic"]), np.maximum(w["Real"], w["Synthetic"])
        E[f"comb_corr_{meth}_between"] = int(((w["Combined"] >= lo_ - 1e-9) & (w["Combined"] <= hi_ + 1e-9)).sum()); E[f"corr_{meth}_pairs"] = int(len(w))
        E[f"syn_corr_{meth}_weaker"] = int((w["Synthetic"].abs() < w["Real"].abs()).sum())
    oc = L.outlier_comparison(L.combine(r, s))
    E["pv_out_real"], E["pv_out_syn"], E["pv_out_comb"] = (float(oc.loc["PageValues", k]) for k in
        ("outliers Real (own fences) %", "outliers Synthetic (own fences) %", "outliers Combined (combined fences) %"))
    E["syn_max_below_real"] = int((oc["max Synthetic"] < oc["max Real"]).sum()); E["comb_max_equals_real"] = int((oc["max Combined"] == oc["max Real"]).sum())
    E["pv_syn_beyond_real_fence"] = float(oc.loc["PageValues", "synthetic rows beyond Real fences %"])
    mt = {k: L.monthly_table(dfs[k]) for k in SC}
    for k, key in (("Real", "real"), ("Synthetic", "syn"), ("Combined", "comb")):
        E[f"rate_min_{key}"], E[f"rate_max_{key}"] = float(mt[k].rate.min()), float(mt[k].rate.max())
        E[f"rate_max_month_{key}"] = mt[k].rate.idxmax()
    E["month_n_min_real"] = int(mt["Real"].n.min()); E["month_n_min_real_month"] = mt["Real"].n.idxmin()
    E["month_feb_ci_real"] = (float(mt["Real"].loc["Feb", "lo"]), float(mt["Real"].loc["Feb", "hi"]))
    E["month_syn_outside_real_ci"] = int(((mt["Synthetic"].rate < mt["Real"].lo) | (mt["Synthetic"].rate > mt["Real"].hi)).sum())
    E["month_syn_outside_list"] = [m for m in mt["Real"].index if (mt["Synthetic"].rate[m] < mt["Real"].lo[m]) or (mt["Synthetic"].rate[m] > mt["Real"].hi[m])]
    E["month_rate_spearman_syn"] = float(stats.spearmanr(mt["Real"].rate, mt["Synthetic"].rate)[0])
    E["month_rate_spearman_comb"] = float(stats.spearmanr(mt["Real"].rate, mt["Combined"].rate)[0])
    E["month_share_gap_max"] = float((mt["Synthetic"].share - mt["Real"].share).abs().max())
    de = pd.DataFrame({k: driver_effects(dfs[k]) for k in SC}); E["driver_table"] = de
    E["driver_spearman_syn"] = float(stats.spearmanr(de.Real, de.Synthetic)[0]); E["driver_spearman_comb"] = float(stats.spearmanr(de.Real, de.Combined)[0])
    num_i = [i for i in de.index if "rank-biserial" in i]; cat_i = [i for i in de.index if "Cramer" in i]
    for key, col in (("syn", "Synthetic"), ("comb", "Combined")):
        E[f"driver_spearman_num_{key}"] = float(stats.spearmanr(de.loc[num_i, "Real"], de.loc[num_i, col])[0])
        E[f"driver_spearman_cat_{key}"] = float(stats.spearmanr(de.loc[cat_i, "Real"], de.loc[cat_i, col])[0])
    ratio = (de.Synthetic / de.Real); E["driver_ratio_min"] = float(ratio.min()); E["driver_ratio_max"] = float(ratio.max())
    E["driver_ratio_min_name"] = ratio.idxmin(); E["driver_ratio_max_name"] = ratio.idxmax()
    E["driver_top"] = {k: de[k].idxmax().split(" (")[0] for k in SC}
    for k, key in (("Real", "real"), ("Synthetic", "syn"), ("Combined", "comb")):
        E[f"weekend_p_{key}"] = float(L.cramers_v(dfs[k].Weekend, dfs[k].Revenue)[1])
        E[f"purchase_rate_{key}"] = float(100 * (dfs[k].Revenue == "Purchase").mean())
    return E


def summary_matrix(E):
    f = lambda x, d=1: f"{x:.{d}f}"
    rows = [
        ("Mean",
         f"Within 8% of the real mean for {E['syn_mean_within8']} of 14 columns ({E['thr_mean'][5]} / {E['thr_mean'][10]} / {E['thr_mean'][15]} columns within 5% / 10% / 15%); largest gap {E['syn_mean_diff_max_col']} ({f(E['syn_mean_diff_max'])}%).",
         f"Combined mean is the average of the two sources, so it lies between them: within 8% of real for {E['comb_mean_within8']} of 14 columns ({E['thr_mean_comb'][5]} / {E['thr_mean_comb'][10]} / {E['thr_mean_comb'][15]} within 5% / 10% / 15%), largest gap {E['comb_mean_diff_max_col']} ({f(E['comb_mean_diff_max'])}%)."),
        ("Median",
         f"Within 10% of the real median (or both 0) for {E['syn_median_close']} of 14 columns; the exceptions are {', '.join(E['syn_median_far'])}.",
         f"Within 10% of the real median (or both 0) for {E['comb_median_close']} of 14 columns; the exceptions are {', '.join(E['comb_median_far']) if E['comb_median_far'] else 'none'}."),
        ("Variance (standard deviation)",
         f"Std ratio Synthetic/Real from {f(E['syn_std_ratio_min'], 2)} ({E['syn_std_ratio_min_col']}) to {f(E['syn_std_ratio_max'], 2)} ({E['syn_std_ratio_max_col']}).",
         f"Std ratio Combined/Real from {f(E['comb_std_ratio_min'], 2)} ({E['comb_std_ratio_min_col']}) to {f(E['comb_std_ratio_max'], 2)} ({E['comb_std_ratio_max_col']}): spread is lowered in sparse columns and raised in browsing columns."),
        ("Distribution shape",
         f"KS above 0.05 in {E['syn_ks_gt05']} of 14 columns (above 0.03 / 0.10: {E['thr_ks'][0.03]} / {E['thr_ks'][0.10]}); mean KS {f(E['syn_ks_mean'], 3)}; largest {E['syn_ks_max_col']} ({f(E['syn_ks_max'], 3)}).",
         f"KS to Real above 0.05 in {E['comb_ks_gt05']} of 14 columns (above 0.03 / 0.10: {E['thr_ks_comb'][0.03]} / {E['thr_ks_comb'][0.10]}); mean KS {f(E['comb_ks_mean'], 3)}; largest {E['comb_ks_max_col']} ({f(E['comb_ks_max'], 3)}): about half the synthetic distance."),
        ("Categorical proportions",
         f"Cramér's V between source and column {f(E['cat_V_min'], 3)} to {f(E['cat_V_max'], 3)}; largest share gap {f(E['cat_gap_max'], 2)} pp ({E['cat_gap_max_col']}).",
         f"Largest share gap of Combined to Real {f(E['comb_cat_gap_max'], 2)} pp: shares are the average of two almost equal sources."),
        ("Correlation",
         f"Mean |Δ Pearson| over 91 pairs {f(E['syn_corr_pearson_mean'], 3)} (max {f(E['syn_corr_pearson_max'], 2)}, {E['syn_corr_pearson_max_pair']}); |r| is smaller than in the real data for {E['syn_corr_pearson_weaker']} of 91 pairs; sign changes: {E['syn_corr_pearson_signflip']} of {E['corr_pearson_strong_pairs']} pairs with real |r| ≥ 0.1, {E['syn_corr_pearson_signflip_r30']} of {E['corr_pearson_strong_pairs_r30']} with real |r| ≥ 0.3.",
         f"Mean |Δ Pearson| to Real {f(E['comb_corr_pearson_mean'], 3)} (max {f(E['comb_corr_pearson_max'], 2)}); {E['comb_corr_pearson_signflip']} sign changes (|r| ≥ 0.1). Combined lies between Real and Synthetic for {E['comb_corr_pearson_between']} of 91 Pearson pairs, not for all, because pooling two samples does not average correlations."),
        ("Outliers and extremes",
         f"PageValues outliers {f(E['pv_out_real'])}% real, {f(E['pv_out_syn'])}% synthetic; synthetic maximum below the real maximum in {E['syn_max_below_real']} of 14 columns.",
         f"Combined maximum equals the real maximum in {E['comb_max_equals_real']} of 14 columns (no new extremes); PageValues outliers {f(E['pv_out_comb'])}% (a mixture)."),
        ("Trends (monthly)",
         f"Monthly purchase rate {f(E['rate_min_real'])}% to {f(E['rate_max_real'])}% real ({E['rate_max_month_real']} peak; {E['month_n_min_real_month']} has only {E['month_n_min_real']} sessions, 95% interval {f(E['month_feb_ci_real'][0])}% to {f(E['month_feb_ci_real'][1])}%) against {f(E['rate_min_syn'])}% to {f(E['rate_max_syn'])}% synthetic; the synthetic rate lies outside the real 95% interval in {E['month_syn_outside_real_ci']} of 10 months; Spearman correlation of the ten monthly rates {f(E['month_rate_spearman_syn'], 2)}; month shares differ by at most {f(E['month_share_gap_max'], 2)} pp.",
         f"Combined range {f(E['rate_min_comb'])}% to {f(E['rate_max_comb'])}%; the seasonal pattern is kept in direction (rank correlation with real {f(E['month_rate_spearman_comb'], 2)}, which is high largely because half of the rows are real) and understated in size."),
        ("Analytical patterns (drivers of purchase)",
         f"Purchase rate {f(E['purchase_rate_real'], 2)}% real, {f(E['purchase_rate_syn'], 2)}% synthetic; the ranking of the 11 drivers (rank-biserial and Cramér's V values ranked together) has a Spearman correlation of {f(E['driver_spearman_syn'], 2)} between Real and Synthetic ({f(E['driver_spearman_num_syn'], 2)} within the 6 numerical drivers); top driver {E['driver_top']['Real']} (real) against {E['driver_top']['Synthetic']} (synthetic); weekend p = {E['weekend_p_real']:.4f} real, {E['weekend_p_syn']:.2f} synthetic.",
         f"Combined purchase rate {f(E['purchase_rate_comb'], 2)}%; driver ranking correlation with real {f(E['driver_spearman_comb'], 2)}; effects between the two sources; weekend p = {E['weekend_p_comb']:.3f}."),
    ]
    return pd.DataFrame(rows, columns=["Aspect", "Real vs. Synthetic", "Effect of the Combined dataset"]).set_index("Aspect")


# ---------------------------------------------------------------------------------------------- Power BI export tables
def scenario_dimension():
    """Small dimension table for the scenario slicer."""
    return pd.DataFrame({"scenario": SC, "scenario_order": [1, 2, 3],
                         "description": ["Real cleaned sessions (12,199 rows)", "Synthetic sessions from the Gaussian copula (12,199 rows)",
                                         "Real and synthetic rows pooled (24,398 rows)"]})


def rows_by_scenario(dfs):
    """Row-level table with a ``scenario`` column. Combined = all rows of Real and Synthetic, so a single-select
    slicer on ``scenario`` shows exactly one of the three datasets and the same DAX measures work for each."""
    out = []
    for s in SC:
        d = dfs[s].copy(); d.insert(0, "scenario", s); out.append(d)
    return pd.concat(out, ignore_index=True)


def box_stats(dfs):
    """Five-number summary plus fences for every numerical column and scenario (for box-style visuals)."""
    rows = []
    for c in L.NUM:
        for s in SC:
            x = dfs[s][c]; q1, med, q3 = x.quantile([.25, .5, .75]); i = q3 - q1
            rows.append({"column": c, "scenario": s, "min": x.min(), "q1": q1, "median": med, "q3": q3, "max": x.max(),
                         "lower_fence": max(x.min(), q1 - 1.5 * i), "upper_fence": min(x.max(), q3 + 1.5 * i),
                         "outlier_pct": 100 * ((x < q1 - 1.5 * i) | (x > q3 + 1.5 * i)).mean()})
    return pd.DataFrame(rows)


def histogram_bins(dfs, S):
    """Percentage of each scenario's rows in the shared bins of the comparison histograms (values above the real
    99th percentile fall in the last bin, as in the Python figures)."""
    rows = []
    for c in L.NUM:
        e = S["bins"][c]
        for s in SC:
            h = L._hist_pct(dfs[s][c].clip(upper=S["hi"][c]), e)
            for j, v in enumerate(h):
                rows.append({"column": c, "scenario": s, "bin_index": j + 1, "bin_left": e[j], "bin_right": e[j + 1],
                             "bin_label": f"{e[j]:.3g} to {e[j + 1]:.3g}", "pct_of_rows": float(v)})
    return pd.DataFrame(rows)
