"""Phase 7: advanced analysis of the real and the synthetic data (hypothesis tests, PCA, feature importance, clustering).

All functions take the cleaned real table and the synthetic table. Random operations use the fixed ``SEED``.
Base numerical columns are the ten columns that the generator models; the four derived columns
(Total_Pages, Total_Duration, Avg_Time_Per_Page, Product_Page_Share) are left out of PCA, clustering and the
prediction task, because they are exact functions of the base columns and would count the same information twice.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

import scenario_lib as L

SEED = 42
COL = L.SCENARIO_COLOR
BASE_NUM = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
# Quarter and Session_Length_Band are derived from Month and Total_Duration, so they are not used as features
BASE_CAT = ["Month", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region", "TrafficType"]


# ------------------------------------------------------------------------------------------ 1. hypothesis tests
def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def assumption_table(real, syn):
    """Normality (Shapiro-Wilk on a seeded sample of 5,000 rows), skewness and equal-variance test (Levene) per column."""
    rng = np.random.default_rng(SEED); rows = []
    for c in L.NUM:
        r = real[c].to_numpy(); s = syn[c].to_numpy()
        sw_r = stats.shapiro(rng.choice(r, 5000, replace=False)).pvalue; sw_s = stats.shapiro(rng.choice(s, 5000, replace=False)).pvalue
        rows.append({"column": c, "skew real": stats.skew(r), "skew synthetic": stats.skew(s),
                     "Shapiro p real": sw_r, "Shapiro p synthetic": sw_s,
                     "Levene p": stats.levene(r, s, center="median").pvalue, "std ratio": s.std(ddof=1) / r.std(ddof=1)})
    return pd.DataFrame(rows).set_index("column")


def numeric_tests(real, syn):
    """Welch t-test and Mann-Whitney U test, real vs synthetic, with effect sizes and Holm-adjusted p-values.
    The Mann-Whitney test is the primary test, because no column is normal (see ``assumption_table``)."""
    rows = []
    for c in L.NUM:
        r, s = real[c], syn[c]
        t = stats.ttest_ind(r, s, equal_var=False)
        u = stats.mannwhitneyu(r, s, alternative="two-sided")
        sd = np.sqrt((r.var() + s.var()) / 2)
        rows.append({"column": c, "Welch t": t.statistic, "Welch p": t.pvalue, "Cohen d": (r.mean() - s.mean()) / sd if sd else np.nan,
                     "U": u.statistic, "MWU p": u.pvalue, "rank-biserial": 2 * u.statistic / (len(r) * len(s)) - 1,
                     "KS": stats.ks_2samp(r, s).statistic})
    d = pd.DataFrame(rows).set_index("column")
    d["MWU p (Holm)"] = holm(d["MWU p"]); d["Welch p (Holm)"] = holm(d["Welch p"])
    d["significant (Holm, 5%)"] = d["MWU p (Holm)"] < 0.05
    d["non-negligible"] = (d["Cohen d"].abs() >= 0.2) | (d["rank-biserial"].abs() >= 0.1)
    return d


def categorical_tests(real, syn):
    """Chi-square test of homogeneity (real vs synthetic) per categorical column, Cramer's V and Holm-adjusted p-values."""
    rows = []
    for c in L.CAT:
        both = pd.concat([real[c], syn[c]], ignore_index=True); src = np.r_[np.zeros(len(real)), np.ones(len(syn))]
        ct = pd.crosstab(both, src); chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
        # robustness: pool the levels with fewer than 50 rows (both sources together) into one level, then test again
        tot = both.value_counts(); rare = tot[tot < 50].index; pooled = both.where(~both.isin(rare), "Other (pooled)") if len(rare) > 1 else both
        p_pooled = stats.chi2_contingency(pd.crosstab(pooled, src), correction=False)[1]
        rows.append({"column": c, "levels": ct.shape[0], "chi2": chi2, "dof": dof, "p": p, "p (rare levels pooled)": p_pooled,
                     "Cramer's V": np.sqrt(chi2 / (ct.values.sum() * min(ct.shape[1] - 1, ct.shape[0] - 1))),
                     "min expected count": exp.min()})
    d = pd.DataFrame(rows).set_index("column")
    d["p (Holm)"] = holm(d["p"]); d["significant (Holm, 5%)"] = d["p (Holm)"] < 0.05; d["non-negligible"] = d["Cramer's V"] >= 0.1
    return d


def fig_tests(nt, ct, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    order = nt["rank-biserial"].abs().sort_values().index
    y = np.arange(len(order)); ax = axes[0]
    ax.barh(y - 0.2, nt.loc[order, "rank-biserial"].abs(), 0.4, color=COL["Real"], label="|rank-biserial| (Mann-Whitney)")
    ax.barh(y + 0.2, nt.loc[order, "Cohen d"].abs(), 0.4, color=COL["Synthetic"], label="|Cohen d| (Welch t-test)")
    ax.axvline(0.1, color="k", ls=":", lw=1); ax.axvline(0.2, color="k", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8); ax.set_xlabel("effect size (real vs synthetic)")
    ax.set_title("Numerical columns: size of the difference (dotted 0.1, dashed 0.2)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    order = ct["Cramer's V"].sort_values().index; ax = axes[1]
    ax.barh(order, ct.loc[order, "Cramer's V"], color=COL["Real"]); ax.axvline(0.1, color="k", ls=":", lw=1)
    ax.set_xlabel("Cramér's V (real vs synthetic)"); ax.set_title("Categorical columns (dotted 0.1: small effect)"); ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); L.save(fig, path)


# ------------------------------------------------------------------------------------------ 2. PCA
def prep_matrix(df, scaler=None):
    """log1p of the ten base numerical columns (they are right-skewed and zero-heavy), then standardisation."""
    X = np.log1p(df[BASE_NUM].to_numpy(float))
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def pca_analysis(real, syn, n=5):
    Xr, sc = prep_matrix(real); Xs, _ = prep_matrix(syn, sc)            # both scaled with the REAL mean and std
    pr = PCA(n_components=n, random_state=SEED).fit(Xr)
    ps = PCA(n_components=n, random_state=SEED).fit(prep_matrix(syn)[0])   # PCA fitted on the synthetic data alone
    Zr, Zs = pr.transform(Xr), pr.transform(Xs)
    sim = []
    for k in range(n):
        a, b = pr.components_[k], ps.components_[k]
        sim.append(abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    var = pd.DataFrame({"component": [f"PC{k + 1}" for k in range(n)], "explained variance real": pr.explained_variance_ratio_,
                        "explained variance synthetic (own PCA)": ps.explained_variance_ratio_, "loading cosine similarity": sim,
                        "KS of scores (synthetic projected on real PCA)": [stats.ks_2samp(Zr[:, k], Zs[:, k]).statistic for k in range(n)],
                        "mean score real": Zr.mean(0), "mean score synthetic": Zs.mean(0),
                        "std score real": Zr.std(0, ddof=1), "std score synthetic": Zs.std(0, ddof=1)})
    load = pd.DataFrame(pr.components_.T, index=BASE_NUM, columns=[f"PC{k + 1}" for k in range(n)])
    load_s = pd.DataFrame(ps.components_.T, index=BASE_NUM, columns=[f"PC{k + 1}" for k in range(n)])
    return {"Zr": Zr, "Zs": Zs, "var": var, "loadings_real": load, "loadings_syn": load_s, "pca_real": pr}


def fig_pca(res, path, n=3000):
    rng = np.random.default_rng(SEED); Zr, Zs = res["Zr"], res["Zs"]
    ir = rng.choice(len(Zr), n, replace=False); is_ = rng.choice(len(Zs), n, replace=False)
    lim = np.percentile(np.r_[Zr[:, :2].ravel(), Zs[:, :2].ravel()], [0.5, 99.5]); ev = res["pca_real"].explained_variance_ratio_
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    axes[0].scatter(Zr[ir, 0], Zr[ir, 1], s=5, alpha=0.35, color=COL["Real"]); axes[0].set_title("Real (3,000 random points)")
    axes[1].scatter(Zs[is_, 0], Zs[is_, 1], s=5, alpha=0.35, color=COL["Synthetic"]); axes[1].set_title("Synthetic (3,000 random points)")
    axes[2].scatter(Zr[ir, 0], Zr[ir, 1], s=5, alpha=0.3, color=COL["Real"], label="Real")
    axes[2].scatter(Zs[is_, 0], Zs[is_, 1], s=5, alpha=0.3, color=COL["Synthetic"], label="Synthetic"); axes[2].set_title("Overlay")
    axes[2].legend(frameon=False, markerscale=3)
    for ax, Z in ((axes[0], Zr), (axes[1], Zs)):          # share of ALL rows in the dense left group (overplotted in the scatter)
        ax.text(0.03, 0.04, f"{100 * (Z[:, 0] < -4.5).mean():.2f}% of all rows have PC1 < -4.5", transform=ax.transAxes, fontsize=8)
    for ax in axes:
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of variance)")
    axes[0].set_ylabel(f"PC2 ({100 * ev[1]:.1f}% of variance)")
    fig.suptitle("PCA of the ten base numerical columns (log1p, standardised with the real mean and std; PCA fitted on the real data)", y=1.02, fontsize=11)
    fig.tight_layout(); L.save(fig, path)


def fig_pca_variance(res, path):
    v = res["var"]; x = np.arange(len(v)); fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(x - 0.2, v["explained variance real"], 0.4, color=COL["Real"], label="Real"); axes[0].bar(x + 0.2, v["explained variance synthetic (own PCA)"], 0.4, color=COL["Synthetic"], label="Synthetic")
    axes[0].set_xticks(x); axes[0].set_xticklabels(v.component); axes[0].set_ylabel("share of variance"); axes[0].set_title("Explained variance (each PCA fitted on its own data)"); axes[0].legend(frameon=False)
    axes[1].bar(x, v["loading cosine similarity"], color="#7A7A7A"); axes[1].set_ylim(0, 1.05); axes[1].set_xticks(x); axes[1].set_xticklabels(v.component)
    axes[1].set_title("Similarity of the loadings (|cosine|) of real and synthetic PCA"); fig.tight_layout(); L.save(fig, path)


def two_sample_classifier(real, syn):
    """Classifier two-sample test: can a model tell real rows from synthetic rows? Cross-validated ROC AUC
    (0.5 = indistinguishable, 1 = perfectly separable). All 24 columns (derived ones included) are used."""
    d = pd.concat([real.assign(_y=0), syn.assign(_y=1)], ignore_index=True); y = d.pop("_y").to_numpy()
    X = pd.get_dummies(d.drop(columns=["Revenue"]).assign(Revenue=d.Revenue), dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {}
    for name, cols in (("all columns", X.columns), ("numerical columns only", [c for c in X.columns if c in L.NUM]),
                       ("categorical columns only", [c for c in X.columns if c not in L.NUM])):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
        p = cross_val_predict(clf, X[cols], y, cv=cv, method="predict_proba")[:, 1]
        out[name] = roc_auc_score(y, p)
    return pd.Series(out, name="ROC AUC (5-fold CV)")


# ------------------------------------------------------------------------------------------ 3. feature importance
def design_matrix(df, columns=None):
    X = pd.get_dummies(df[BASE_NUM + BASE_CAT], columns=BASE_CAT, dtype=float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def fig_importance(imp, perf, path):
    names = ["Real", "Synthetic (fitted on training rows only)", "Real + Synthetic (fitted on training rows only)"]
    short = {names[0]: "trained on Real", names[1]: "trained on Synthetic", names[2]: "trained on Real + Synthetic"}
    colors = {names[0]: COL["Real"], names[1]: COL["Synthetic"], names[2]: COL["Combined"]}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    order = imp["Real"].sort_values().index; y = np.arange(len(order))
    for k, name in enumerate(names):
        axes[0].barh(y + (k - 1) * 0.27, imp.loc[order, name], 0.27, color=colors[name], label=short[name])
    axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=8); axes[0].set_xlabel("drop in ROC AUC when the attribute is shuffled (real test set)")
    axes[0].set_title("Grouped permutation importance for predicting Revenue"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    p = perf.drop(index="baseline: majority class"); lab = ["Real", "Synthetic", "Real + Synthetic", "Synthetic (main;\ngenerator saw test rows)"]
    cols = [COL["Real"], COL["Synthetic"], COL["Combined"], "#F2D3A8"]
    axes[1].bar(range(len(p)), p["ROC AUC"], color=cols); axes[1].set_xticks(range(len(p))); axes[1].set_xticklabels(lab, fontsize=7.5)
    axes[1].set_ylim(0.5, 1.0); axes[1].set_ylabel("ROC AUC on real test data"); axes[1].set_title("Prediction quality (train on ..., test on real)")
    for i, v in enumerate(p["ROC AUC"]): axes[1].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); L.save(fig, path)


# ------------------------------------------------------------------------------------------ 4. clustering
def silhouette_curve(real, syn, ks=range(2, 9), n=4000):
    rng = np.random.default_rng(SEED); rows = []
    for name, d in (("Real", real), ("Synthetic", syn)):
        X, _ = prep_matrix(d); idx = rng.choice(len(X), n, replace=False)
        for k in ks:
            lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
            rows.append({"dataset": name, "k": k, "silhouette": silhouette_score(X[idx], lab[idx])})
    return pd.DataFrame(rows)


def cluster_analysis(real, syn, k):
    """K-Means with k clusters. (a) Fitted on each dataset separately, with the clusters of the synthetic fit matched to the real
    ones by centroid distance. (b) Fitted on the real data and applied to the synthetic data. Everything is in the real scaling."""
    Xr, sc = prep_matrix(real); Xs, _ = prep_matrix(syn, sc)
    kr = KMeans(k, n_init=10, random_state=SEED).fit(Xr); ks = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
    cost = ((kr.cluster_centers_[:, None, :] - ks.cluster_centers_[None, :, :]) ** 2).sum(-1) ** 0.5
    ri, ci = linear_sum_assignment(cost); match = dict(zip(ri, ci))           # real cluster i <-> synthetic cluster match[i]
    lab_r = kr.labels_; lab_s_own = np.array([{v: u for u, v in match.items()}[l] for l in ks.labels_]); lab_s_applied = kr.predict(Xs)
    prof = lambda d, lab: d.assign(cluster=lab).groupby("cluster")[BASE_NUM].mean()
    sizes = pd.DataFrame({"Real (own fit)": pd.Series(lab_r).value_counts(normalize=True).sort_index() * 100,
                          "Synthetic (own fit, matched)": pd.Series(lab_s_own).value_counts(normalize=True).sort_index() * 100,
                          "Synthetic (real fit applied)": pd.Series(lab_s_applied).value_counts(normalize=True).sort_index() * 100})
    buy = lambda d, lab: (d.assign(cluster=lab).groupby("cluster").Revenue.apply(lambda s: 100 * (s == "Purchase").mean()))
    purchase = pd.DataFrame({"Real (own fit)": buy(real, lab_r), "Synthetic (own fit, matched)": buy(syn, lab_s_own), "Synthetic (real fit applied)": buy(syn, lab_s_applied)})
    cdist = pd.Series({f"cluster {i}": cost[i, match[i]] for i in range(k)})
    return {"kr": kr, "ks": ks, "sizes": sizes, "purchase": purchase, "centroid_dist": cdist, "match": match,
            "profile_real": prof(real, lab_r), "profile_syn": prof(syn, lab_s_own), "profile_syn_applied": prof(syn, lab_s_applied),
            "ari_applied_vs_own": adjusted_rand_score(lab_s_own, lab_s_applied), "Zr": Xr, "Zs": Xs, "lab_r": lab_r, "lab_s": lab_s_own,
            "centroids_real": pd.DataFrame(kr.cluster_centers_, columns=BASE_NUM), "centroids_syn": pd.DataFrame(ks.cluster_centers_[[match[i] for i in range(k)]], columns=BASE_NUM)}


def fig_silhouette(sil, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for name in ("Real", "Synthetic"):
        t = sil[sil.dataset == name]; ax.plot(t.k, t.silhouette, marker="o", color=COL[name], label=name)
    ax.set_xlabel("number of clusters k"); ax.set_ylabel("silhouette (4,000-row sample)"); ax.set_title("Choice of k"); ax.legend(frameon=False); fig.tight_layout(); L.save(fig, path)


def fig_clusters(res, path):
    k = len(res["sizes"]); fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    vmax = float(np.abs(np.r_[res["centroids_real"].to_numpy().ravel(), res["centroids_syn"].to_numpy().ravel()]).max())
    cr, cs = res["centroids_real"].T, res["centroids_syn"].T; both = pd.concat([cr.add_prefix("Real "), cs.add_prefix("Synth ")], axis=1)
    order = [c for i in range(k) for c in (f"Real {i}", f"Synth {i}")]
    sns.heatmap(both[order], cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, annot=True, fmt=".1f", annot_kws={"fontsize": 6.5}, cbar=False, ax=axes[0])
    axes[0].set_title("Cluster centres (standardised log1p, real scaling); Real and matched Synthetic side by side"); axes[0].tick_params(labelsize=7); axes[0].set_title(axes[0].get_title(), fontsize=8)
    s = res["sizes"]; x = np.arange(k); w = 0.27
    for j, (c, name) in enumerate(zip(s.columns, ["Real", "Synthetic", "Synthetic"])):
        axes[1].bar(x + (j - 1) * w, s[c], w, color=[COL["Real"], COL["Synthetic"], "#F2B27A"][j], label=c)
    axes[1].set_xticks(x); axes[1].set_xlabel("cluster"); axes[1].set_ylabel("% of rows"); axes[1].set_title("Cluster sizes"); axes[1].legend(frameon=False, fontsize=7)
    p = res["purchase"]
    for j, c in enumerate(p.columns):
        axes[2].bar(x + (j - 1) * w, p[c], w, color=[COL["Real"], COL["Synthetic"], "#F2B27A"][j])
    axes[2].set_xticks(x); axes[2].set_xlabel("cluster"); axes[2].set_ylabel("purchase rate (%)"); axes[2].set_title("Purchase rate per cluster")
    fig.tight_layout(); L.save(fig, path)


# ------------------------------------------------------------------------------------------ generator refit (hold-out safe)
_GEN_NUM = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues"]
_GEN_CAT = ["Month", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region", "TrafficType", "Revenue"]
_SPECIAL = "SpecialDay"
_COUNT_DUR = [("Administrative", "Administrative_Duration"), ("Informational", "Informational_Duration"),
              ("ProductRelated", "ProductRelated_Duration")]
_GENERATED = _GEN_NUM + [_SPECIAL] + _GEN_CAT


def generate_from(train, n, seed=SEED):
    """Re-run the project's generator (notebook 02, final configuration D: Gaussian copula with a hurdle for zeros, log scale
    for the positive parts, zero indicator modelled as a column) but fitted on the rows of ``train`` only, with the same repair
    and validity rules and seed. Returns the base columns (derived columns are not recomputed: they are not used in Task 3).
    Used so that the synthetic training data of the prediction task has never seen the real test rows."""
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer
    from rdt.transformers import FloatFormatter
    real_max = train[_GEN_NUM].max(); levels = {c: set(train[c].unique()) for c in _GEN_CAT + [_SPECIAL]}

    t = train[_GENERATED].copy(); t[_SPECIAL] = t[_SPECIAL].astype(str)
    for c in _GEN_NUM:
        t[c] = t[c].where(t[c] != 0, np.nan)
        t[c] = np.log(t[c])
    meta = Metadata.detect_from_dataframes({"t": t})
    for c in _GEN_NUM: meta.update_column(c, sdtype="numerical")
    for c in _GEN_CAT + [_SPECIAL]: meta.update_column(c, sdtype="categorical")
    np.random.seed(seed)
    syn = GaussianCopulaSynthesizer(meta, enforce_min_max_values=True, enforce_rounding=False, default_distribution="truncnorm",
                                    numerical_distributions={c: "norm" for c in _GEN_NUM})
    syn.auto_assign_transformers({"t": t})
    syn.update_transformers({c: FloatFormatter(learn_rounding_scheme=False, enforce_min_max_values=True,
                                               missing_value_replacement="mean", missing_value_generation="from_column") for c in _GEN_NUM})
    syn.fit({"t": t})

    def repair(a):
        a = a.copy()
        for c in ("Administrative", "Informational", "ProductRelated"): a[c] = a[c].round()
        a[_GEN_NUM] = a[_GEN_NUM].fillna(0).clip(lower=0).clip(upper=real_max, axis=1)
        for k, d in _COUNT_DUR:
            a.loc[(a[k] == 0) & (a[d] > 0), d] = 0
        for c in ("Administrative", "Informational", "ProductRelated"): a[c] = a[c].astype(int)
        a[_SPECIAL] = a[_SPECIAL].astype(float)
        return a

    def valid(a):
        ok = a[["Administrative", "Informational", "ProductRelated"]].sum(axis=1) >= 1
        ok &= (a[["BounceRates", "ExitRates"]] <= 1).all(axis=1)
        for c in _GEN_CAT: ok &= a[c].isin(levels[c])
        return ok & a[_SPECIAL].isin([float(x) for x in levels[_SPECIAL]])

    kept, need, tries = [], n, 0
    while need > 0 and tries < 20:
        tries += 1
        out = syn.sample(int(need * 1.2) + 50); out = (out["t"] if isinstance(out, dict) else out).copy()
        for c in _GEN_NUM: out[c] = np.exp(out[c])
        raw = repair(out); kept.append(raw[valid(raw)]); need = n - sum(len(k) for k in kept)
    return pd.concat(kept).iloc[:n].reset_index(drop=True)


# ------------------------------------------------------------------------------------------ hold-out safe feature importance
def grouped_permutation_importance(model, X, y, groups, repeats=10, seed=SEED):
    """Permutation importance in which all columns of a group (the dummy columns of one categorical attribute) are shuffled
    together, i.e. whole rows of the group are permuted, so that no impossible combination of dummies is created.
    Returns the mean drop in ROC AUC per group."""
    rng = np.random.default_rng(seed); Xv = X.to_numpy(); base = roc_auc_score(y, model.predict_proba(X)[:, 1]); out = {}
    cols = {g: [X.columns.get_loc(c) for c in cs] for g, cs in groups.items()}
    for g, idx in cols.items():
        drops = []
        for _ in range(repeats):
            Xp = Xv.copy(); perm = rng.permutation(len(Xv)); Xp[:, idx] = Xv[perm][:, idx]
            drops.append(base - roc_auc_score(y, model.predict_proba(pd.DataFrame(Xp, columns=X.columns))[:, 1]))
        out[g] = float(np.mean(drops))
    return pd.Series(out)


def paired_bootstrap_auc(y, p_a, p_b, n_boot=1000, seed=SEED):
    """Paired bootstrap over the test rows: mean and 95% interval of AUC(a) - AUC(b) on the same resampled rows."""
    rng = np.random.default_rng(seed); y = np.asarray(y); d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        d.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    d = np.array(d); return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def importance_analysis2(real, syn_main, drop=(), repeats=10, n_boot=1000):
    """Task 3, hold-out safe version. 70% of the real rows train, 30% test (stratified, seed 42). A second synthetic dataset is
    generated with the project's generator fitted on the 70% training rows only (same size as the main synthetic data), so that
    no model has seen the test rows, directly or through the generator. The main synthetic dataset (fitted on all real rows) is
    kept as a fifth row for comparison. Importance: grouped permutation importance on the real test set."""
    y_of = lambda d: (d.Revenue == "Purchase").astype(int).to_numpy()
    tr, te = train_test_split(real, test_size=0.3, stratify=real.Revenue, random_state=SEED)
    syn_tr = generate_from(tr, len(syn_main), SEED)
    cols = design_matrix(real).columns
    keep = [c for c in cols if not any(c == d or c.startswith(d + "_") for d in drop)]
    attrs = [f for f in BASE_NUM + BASE_CAT if f not in drop]
    groups = {f: [c for c in keep if c == f or c.startswith(f + "_")] for f in attrs}
    sets = {"Real": tr, "Synthetic (fitted on training rows only)": syn_tr,
            "Real + Synthetic (fitted on training rows only)": pd.concat([tr, syn_tr], ignore_index=True),
            "Synthetic (main dataset; generator saw the test rows)": syn_main}
    Xte = design_matrix(te, cols)[keep]; yte = y_of(te); res, imp, prob = {}, {}, {}
    for name, d in sets.items():
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(design_matrix(d, cols)[keep], y_of(d))
        p = rf.predict_proba(Xte)[:, 1]; prob[name] = p
        res[name] = {"ROC AUC": roc_auc_score(yte, p), "average precision": average_precision_score(yte, p), "train rows": len(d)}
        imp[name] = grouped_permutation_importance(rf, Xte, yte, groups, repeats)
    res["baseline: majority class"] = {"ROC AUC": 0.5, "average precision": float(yte.mean()), "train rows": 0}
    names = list(sets); ref = "Real"; rows = []
    for nme in names[1:]:
        m, lo, hi = paired_bootstrap_auc(yte, prob[ref], prob[nme], n_boot)
        rows.append({"comparison": f"AUC(Real) - AUC({nme})", "difference": m, "95% lower": lo, "95% upper": hi})
    return pd.DataFrame(res).T, pd.DataFrame(imp), pd.DataFrame(rows).set_index("comparison"), len(te)


def repeated_split_auc(real, n_splits=5):
    """Robustness: repeat split, generator refit and models for five different splits (seeds 42 to 46) and report the test AUC."""
    y_of = lambda d: (d.Revenue == "Purchase").astype(int).to_numpy(); rows = []
    cols = design_matrix(real).columns
    for k in range(n_splits):
        s = SEED + k
        tr, te = train_test_split(real, test_size=0.3, stratify=real.Revenue, random_state=s)
        syn_tr = generate_from(tr, len(real), s); Xte = design_matrix(te, cols); r = {"split seed": s}
        for name, d in (("Real", tr), ("Synthetic", syn_tr), ("Real + Synthetic", pd.concat([tr, syn_tr], ignore_index=True))):
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=s).fit(design_matrix(d, cols), y_of(d))
            r[name] = roc_auc_score(y_of(te), rf.predict_proba(Xte)[:, 1])
        rows.append(r)
    return pd.DataFrame(rows).set_index("split seed")


def shuffled_marginals_auc(real, syn):
    """Classifier two-sample test after shuffling every column independently within each dataset (keeps every column's
    distribution, destroys all relationships between columns). The AUC that remains is what the marginals alone give away."""
    rng = np.random.default_rng(SEED)
    sh = lambda d: pd.DataFrame({c: rng.permutation(d[c].to_numpy()) for c in d.columns})
    return two_sample_classifier(sh(real), sh(syn))["all columns"]
