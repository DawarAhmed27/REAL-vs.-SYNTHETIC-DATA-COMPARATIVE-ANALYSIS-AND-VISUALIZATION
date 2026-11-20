"""Shared analysis code for Scenarios 1-3 (Real, Synthetic, Combined).

The brief requires the *same* analysis, variables, axis scales, bin sizes and plot settings in every
scenario. All of that lives here, so notebooks 04 (real), 05 (synthetic) and 06 (combined) call the same
functions with the same settings object.

Settings are computed once from the real and the synthetic data together (`make_settings`):
  * x-axis limit of a numerical column = the real 99th percentile (the long tail is reported in tables),
  * histogram bins and KDE grids use that limit,
  * y-axis limits are the larger of the real and the synthetic value, so no scenario is clipped.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

SEED = 42
NUM = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
       "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay",
       "Total_Pages", "Total_Duration", "Avg_Time_Per_Page", "Product_Page_Share"]
CAT = ["Month", "Quarter", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region",
       "TrafficType", "Session_Length_Band", "Revenue"]
MONTH_ORDER = ["Feb", "Mar", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LEVEL_ORDER = {"Month": MONTH_ORDER, "Quarter": ["Q1", "Q2", "Q3", "Q4"],
               "VisitorType": ["New_Visitor", "Returning_Visitor", "Other"], "Weekend": ["Weekday", "Weekend"],
               "Session_Length_Band": ["Short", "Medium", "Long"], "Revenue": ["No_Purchase", "Purchase"]}
SCENARIO_COLOR = {"Real": "#4C78A8", "Synthetic": "#F58518", "Combined": "#59A14F"}
REV_COLOR = {"No_Purchase": "#9AA5B1", "Purchase": "#D1495B"}
INT_COUNTS = ["Administrative", "Informational", "ProductRelated", "Total_Pages"]
# the four scatter pairs used for the bivariate analysis (meaningful business pairs)
SCATTER_PAIRS = [("ProductRelated", "ProductRelated_Duration"), ("BounceRates", "ExitRates"),
                 ("ExitRates", "PageValues"), ("Total_Pages", "PageValues")]
BOX_BY_CAT = [("PageValues", "Revenue"), ("ExitRates", "Revenue"), ("Total_Pages", "VisitorType"),
              ("Total_Duration", "Quarter")]
PAIR_VARS = ["PageValues", "ExitRates", "Total_Pages", "Avg_Time_Per_Page"]


def use_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 10, "axes.labelsize": 9, "axes.titleweight": "normal"})


def save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.show()


# ----------------------------------------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------------------------------------
def levels_for(real, c):
    return LEVEL_ORDER.get(c, list(real[c].value_counts().index))


def _hist_pct(x, bins):
    h, _ = np.histogram(x, bins=bins)
    return 100 * h / len(x)


def _kde_curve(x, grid):
    x = np.asarray(x, float); x = x[x > 0]
    if len(x) < 10 or x.std() == 0:
        return None
    return stats.gaussian_kde(x)(grid)


def make_settings(real, syn):
    """Axis limits, bins and level orders shared by every scenario."""
    S = {"hi": {}, "bins": {}, "grid": {}, "hist_ymax": {}, "kde_ymax": {}, "cat_ymax": {}, "levels": {}}
    for c in NUM:
        hi = float(real[c].quantile(0.99)); hi = hi if hi > 0 else float(real[c].max())
        S["hi"][c] = hi
        if c == "SpecialDay":
            bins = np.array([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
        elif c in INT_COUNTS and hi <= 40:
            bins = np.arange(-0.5, hi + 1.5, 1.0)
        else:
            bins = np.linspace(0, hi, 31)
        S["bins"][c] = bins
        S["grid"][c] = np.linspace(0, hi, 200)
        S["hist_ymax"][c] = 1.12 * max(_hist_pct(d[c], bins).max() for d in (real, syn))
        ks = [_kde_curve(d[c][d[c] <= hi], S["grid"][c]) for d in (real, syn)]
        S["kde_ymax"][c] = 1.12 * max(k.max() for k in ks if k is not None) if any(k is not None for k in ks) else 1
    for c in CAT:
        S["levels"][c] = levels_for(real, c)
        S["cat_ymax"][c] = 1.12 * max(100 * d[c].value_counts(normalize=True).max() for d in (real, syn))
    # purchase-rate axis shared by the rate charts
    rates = []
    for d in (real, syn):
        for by in ("VisitorType", "Quarter", "Session_Length_Band", "Month"):
            rates.append(100 * d.groupby(by)["Revenue"].apply(lambda s: (s == "Purchase").mean()).max())
    S["rate_ymax"] = float(np.ceil(1.2 * max(rates) / 5) * 5)
    his = []
    for d in (real, syn):
        dd = d[d.VisitorType != "Other"]
        for by in (["VisitorType", "Weekend"], "Quarter", ["Session_Length_Band", "VisitorType"]):
            his.append(rate_table(dd, by).hi.max())
    S["bar_ymax"] = float(np.ceil(1.1 * max(his) / 5) * 5)
    S["bubble_xmax"] = float(np.ceil(1.15 * max(bubble_table(d).mean_pages.max() for d in (real, syn)) / 10) * 10)
    return S


# ----------------------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------------------
def univariate_numeric(df, S=None):
    rows = []
    for c in NUM:
        x = df[c]; q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
        out = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
        mode = x.mode().iloc[0]
        rows.append({"column": c, "mean": x.mean(), "median": x.median(), "mode": mode, "std": x.std(),
                     "min": x.min(), "Q1": q1, "Q3": q3, "max": x.max(), "skewness": stats.skew(x),
                     "IQR outliers (n)": int(out), "IQR outliers (%)": 100 * out / len(x),
                     "zeros (%)": 100 * (x == 0).mean(),
                     "above axis limit (%)": (100 * (x > S["hi"][c]).mean()) if S else np.nan})
    return pd.DataFrame(rows).set_index("column")


def categorical_freq(df):
    rows = []
    for c in CAT:
        vc = df[c].value_counts()
        for lev, n in vc.items():
            rows.append({"column": c, "level": lev, "count": int(n), "percent": 100 * n / len(df)})
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_table(df, by):
    """Purchase rate (%) with a 95% Wilson interval, per group (by = column or list of columns)."""
    g = df.groupby(by, observed=True)["Revenue"]
    t = pd.DataFrame({"n": g.size(), "purchases": g.apply(lambda s: (s == "Purchase").sum())})
    t["rate"] = 100 * t.purchases / t.n
    ci = [wilson(k, n) for k, n in zip(t.purchases, t.n)]
    t["lo"] = [100 * a for a, _ in ci]; t["hi"] = [100 * b for _, b in ci]
    return t.reset_index()


def cramers_v(a, b):
    ct = pd.crosstab(a, b); chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum(); k = min(ct.shape) - 1
    return chi2, p, dof, np.sqrt(chi2 / (n * k)) if k > 0 else np.nan


def bivariate_tests(df):
    rows = []
    for x, y in SCATTER_PAIRS:
        r, pr = stats.pearsonr(df[x], df[y]); rho, ps = stats.spearmanr(df[x], df[y])
        rows.append({"test": "correlation", "pair": f"{x} ~ {y}", "statistic": f"Pearson r = {r:.3f}; Spearman rho = {rho:.3f}",
                     "effect": rho, "p-value": ps})
    for v in ["PageValues", "ExitRates", "Total_Pages", "Total_Duration"]:
        a = df.loc[df.Revenue == "Purchase", v]; b = df.loc[df.Revenue == "No_Purchase", v]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rb = 2 * u / (len(a) * len(b)) - 1            # rank-biserial correlation (+ means purchases are higher)
        rows.append({"test": "Mann-Whitney U", "pair": f"{v} by Revenue",
                     "statistic": f"median purchase = {a.median():.3g}, no purchase = {b.median():.3g}", "effect": rb, "p-value": p})
    for c in ["VisitorType", "Quarter", "Weekend", "Session_Length_Band", "Month"]:
        chi2, p, dof, v = cramers_v(df[c], df.Revenue)
        rows.append({"test": "chi-square", "pair": f"{c} x Revenue", "statistic": f"chi2 = {chi2:.1f}, dof = {dof}", "effect": v, "p-value": p})
    h, p = stats.kruskal(*[g.Total_Pages.values for _, g in df.groupby("VisitorType")])
    rows.append({"test": "Kruskal-Wallis", "pair": "Total_Pages by VisitorType", "statistic": f"H = {h:.1f}", "effect": h / (len(df) - 1), "p-value": p})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# figures: univariate
# ----------------------------------------------------------------------------------------------------------
def _grid(n_panels, ncol=5, h=3.1, w=3.6):
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * w, nrow * h))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def fig_histograms(df, S, color, title, path):
    fig, axes = _grid(len(NUM))
    for ax, c in zip(axes, NUM):
        x = df[c]
        ax.hist(x.clip(upper=S["hi"][c]), bins=S["bins"][c], weights=np.full(len(x), 100 / len(x)), color=color, alpha=.85,
                edgecolor="white", linewidth=.3)
        ax.set_ylim(0, S["hist_ymax"][c]); ax.set_xlim(S["bins"][c][0], S["bins"][c][-1])
        ax.set_title(c); ax.set_ylabel("% of sessions", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle(title + "\n(values above the real 99th percentile are collected in the last bin)", y=1.0, fontsize=12)
    fig.tight_layout(); save(fig, path)


def fig_kde(df, S, color, title, path):
    cols = [c for c in NUM if c != "SpecialDay"]
    fig, axes = _grid(len(cols))
    for ax, c in zip(axes, cols):
        x = df[c][df[c] <= S["hi"][c]]; k = _kde_curve(x, S["grid"][c])
        if k is not None:
            ax.fill_between(S["grid"][c], k, color=color, alpha=.35); ax.plot(S["grid"][c], k, color=color, lw=1.6)
        ax.set_ylim(0, S["kde_ymax"][c]); ax.set_xlim(0, S["hi"][c])
        ax.set_title(f"{c}  ({100 * (df[c] == 0).mean():.0f}% zeros)"); ax.set_ylabel("density", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle(title + "\n(KDE of the non-zero values up to the real 99th percentile; the share of zeros is in each title)", y=1.0, fontsize=12)
    fig.tight_layout(); save(fig, path)


def fig_boxplots(df, S, color, title, path):
    fig, axes = _grid(len(NUM))
    for ax, c in zip(axes, NUM):
        x = df[c]; q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
        out = 100 * ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean()
        bp = ax.boxplot(x, vert=True, showfliers=False, patch_artist=True, widths=.5)
        bp["boxes"][0].set(facecolor=color, alpha=.7); bp["medians"][0].set(color="black", lw=1.5)
        ax.set_ylim(0, S["hi"][c] * 1.05); ax.set_xticks([])
        if q3 == 0:
            ax.text(.5, .5, "75%+ of values are 0:\nthe box collapses at 0", transform=ax.transAxes, ha="center", fontsize=8, color="#555")
        ax.set_title(f"{c}  (outliers {out:.0f}%)"); ax.tick_params(labelsize=7)
    fig.suptitle(title + "\n(outlier points hidden; the percentage is the share beyond 1.5 x IQR in this scenario)", y=1.0, fontsize=12)
    fig.tight_layout(); save(fig, path)


def fig_categorical_bars(df, S, color, title, path):
    fig, axes = _grid(len(CAT))
    for ax, c in zip(axes, CAT):
        vc = 100 * df[c].value_counts(normalize=True).reindex(S["levels"][c]).fillna(0)
        ax.bar(range(len(vc)), vc.values, color=color, alpha=.9)
        ax.set_xticks(range(len(vc))); ax.set_xticklabels(vc.index, rotation=75 if len(vc) > 4 else 20, ha="right", fontsize=7)
        ax.set_ylim(0, S["cat_ymax"][c]); ax.set_title(c); ax.set_ylabel("% of sessions", fontsize=8); ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(title, y=1.0, fontsize=12); fig.tight_layout(); save(fig, path)


def fig_donuts(df, S, title, path):
    cols = ["Revenue", "VisitorType", "Weekend"]
    pal = sns.color_palette("Set2", 3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, c in zip(axes, cols):
        vc = df[c].value_counts().reindex(S["levels"][c])
        colors = [REV_COLOR[l] for l in vc.index] if c == "Revenue" else pal[:len(vc)]
        ax.pie(vc.values, labels=[f"{l}\n{100 * v / vc.sum():.1f}%" for l, v in vc.items()], colors=colors, startangle=90,
               counterclock=False, wedgeprops=dict(width=.42, edgecolor="white"), textprops={"fontsize": 8})
        ax.set_title(c)
    fig.suptitle(title, y=1.02, fontsize=12); fig.tight_layout(); save(fig, path)


# ----------------------------------------------------------------------------------------------------------
# figures: bivariate
# ----------------------------------------------------------------------------------------------------------
def fig_scatter_pairs(df, S, title, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (x, y) in zip(axes, SCATTER_PAIRS):
        for lev in ("No_Purchase", "Purchase"):
            d = df[df.Revenue == lev]
            ax.scatter(d[x], d[y], s=5, alpha=.25, color=REV_COLOR[lev], label=lev, rasterized=True)
        rho = stats.spearmanr(df[x], df[y])[0]
        ax.set_xlim(0, S["hi"][x] * 1.05); ax.set_ylim(0, S["hi"][y] * 1.05)
        ax.set_xlabel(x, fontsize=8); ax.set_ylabel(y, fontsize=8); ax.set_title(f"{x} vs {y}\nSpearman rho = {rho:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(markerscale=3, fontsize=7)
    fig.suptitle(title, y=1.04, fontsize=12); fig.tight_layout(); save(fig, path)


def fig_box_by_category(df, S, title, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (v, c) in zip(axes, BOX_BY_CAT):
        pal = [REV_COLOR[l] for l in S["levels"][c]] if c == "Revenue" else "Set2"
        sns.boxplot(data=df, x=c, y=v, order=S["levels"][c], hue=c, hue_order=S["levels"][c], palette=pal, ax=ax,
                    showfliers=False, legend=False, width=.6)
        ax.set_ylim(0, S["hi"][v] * 1.05); ax.set_title(f"{v} by {c}", fontsize=9); ax.set_xlabel("")
        ax.tick_params(labelsize=7)
    fig.suptitle(title, y=1.04, fontsize=12); fig.tight_layout(); save(fig, path)


def _rate_bars(ax, t, by, hue, S, levels_by, levels_hue=None, palette=None):
    xs = np.arange(len(levels_by)); hs = levels_hue or [None]; w = .8 / len(hs)
    for i, h in enumerate(hs):
        sub = t if h is None else t[t[hue] == h]
        sub = sub.set_index(by).reindex(levels_by)
        ax.bar(xs + (i - (len(hs) - 1) / 2) * w, sub.rate, w * .95, yerr=[sub.rate - sub.lo, sub.hi - sub.rate],
               color=palette[i] if palette else "#4C78A8", capsize=2, label=h, error_kw={"lw": .8})
    ax.set_xticks(xs); ax.set_xticklabels(levels_by, fontsize=7, rotation=15)
    ax.set_ylim(0, S["bar_ymax"]); ax.set_ylabel("purchase rate (%)", fontsize=8); ax.tick_params(labelsize=7)


def fig_purchase_rate_bars(df, S, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2)); pal = sns.color_palette("Set2", 3)
    df = df[df.VisitorType != "Other"]; vt = [v for v in S["levels"]["VisitorType"] if v != "Other"]
    t = rate_table(df, ["VisitorType", "Weekend"])
    _rate_bars(axes[0], t, "VisitorType", "Weekend", S, vt, S["levels"]["Weekend"], pal[:2])
    axes[0].set_title("Purchase rate by visitor type and weekend", fontsize=9); axes[0].legend(fontsize=7)
    t = rate_table(df, "Quarter"); _rate_bars(axes[1], t, "Quarter", None, S, S["levels"]["Quarter"]); axes[1].set_title("Purchase rate by quarter", fontsize=9)
    t = rate_table(df, ["Session_Length_Band", "VisitorType"])
    _rate_bars(axes[2], t, "Session_Length_Band", "VisitorType", S, S["levels"]["Session_Length_Band"], vt, pal)
    axes[2].set_title("Purchase rate by session length and visitor type", fontsize=9); axes[2].legend(fontsize=7)
    fig.suptitle(title + "\n(error bars: 95% Wilson interval; the 'Other' visitor type, under 1% of sessions, is left out because its intervals are too wide)", y=1.06, fontsize=12); fig.tight_layout(); save(fig, path)


def fig_correlation_heatmaps(df, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    for ax, m, name in zip(axes, ("pearson", "spearman"), ("Pearson", "Spearman")):
        sns.heatmap(df[NUM].corr(method=m), ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                    annot_kws={"size": 6}, cbar_kws={"shrink": .7}, linewidths=.3)
        ax.set_title(name); ax.tick_params(labelsize=7)
    fig.suptitle(title, y=1.0, fontsize=12); fig.tight_layout(); save(fig, path)


# ----------------------------------------------------------------------------------------------------------
# figures: multivariate and trends
# ----------------------------------------------------------------------------------------------------------
def fig_pairplot(df, S, title, path, hue="Revenue", palette=None, n=2500):
    sm = df.sample(n, random_state=SEED)
    palette = palette or REV_COLOR
    g = sns.pairplot(sm, vars=PAIR_VARS, hue=hue, palette=palette, diag_kind="hist", plot_kws=dict(s=7, alpha=.35, rasterized=True),
                     diag_kws=dict(alpha=.5, bins=25, common_norm=False, stat="percent"), height=2.5, corner=False)
    for i, vi in enumerate(PAIR_VARS):
        for j, vj in enumerate(PAIR_VARS):
            ax = g.axes[i, j]; ax.set_xlim(0, S["hi"][vj] * 1.05)
            if i != j:
                ax.set_ylim(0, S["hi"][vi] * 1.05)
    g.fig.suptitle(title + f"\n(random sample of {n:,} sessions, seed {SEED})", y=1.04, fontsize=12)
    save(g.fig, path)


def bubble_table(df):
    t = rate_table(df, ["Month", "VisitorType"])
    m = df.groupby(["Month", "VisitorType"]).Total_Pages.mean().rename("mean_pages").reset_index()
    t = t.merge(m, on=["Month", "VisitorType"]); t["share"] = 100 * t.n / len(df)
    return t[t.n >= 30]          # groups with fewer than 30 sessions are too small for a meaningful mean or rate


def fig_bubble(df, S, title, path):
    t = bubble_table(df); pal = dict(zip(S["levels"]["VisitorType"], sns.color_palette("Set2", 3)))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for v in S["levels"]["VisitorType"]:
        d = t[t.VisitorType == v]
        ax.scatter(d.mean_pages, d.rate, s=d.share * 90, color=pal[v], alpha=.6, edgecolor="white", label=v)
    ax.set_xlim(0, S["bubble_xmax"]); ax.set_ylim(0, S["rate_ymax"] + 10)
    ax.set_xlabel("mean total pages per session"); ax.set_ylabel("purchase rate (%)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=pal[v], markersize=9, alpha=.7, label=v) for v in S["levels"]["VisitorType"]],
              title="visitor type", fontsize=8)
    fig.suptitle(title + "\n(each bubble = one month x visitor type with at least 30 sessions; bubble area = share of all sessions)", y=1.0, fontsize=11)
    fig.tight_layout(); save(fig, path)


def fig_facets(df, S, title, path, n=4000):
    sm = df.sample(n, random_state=SEED)
    g = sns.relplot(data=sm, x="ExitRates", y="PageValues", hue="Revenue", palette=REV_COLOR, col="VisitorType",
                    col_order=S["levels"]["VisitorType"], kind="scatter", s=9, alpha=.35, height=3.6, aspect=1.05, rasterized=True)
    for ax in g.axes.ravel():
        ax.set_xlim(0, S["hi"]["ExitRates"] * 1.05); ax.set_ylim(0, S["hi"]["PageValues"] * 1.05)
    g.fig.suptitle(title + f"\n(random sample of {n:,} sessions, seed {SEED})", y=1.07, fontsize=12)
    save(g.fig, path)


def matrix_table(df):
    t = rate_table(df, ["Quarter", "VisitorType"])
    return t


def fig_rate_matrix(df, S, title, path):
    t = matrix_table(df)
    rate = t.pivot(index="VisitorType", columns="Quarter", values="rate").reindex(S["levels"]["VisitorType"])[S["levels"]["Quarter"]]
    n = t.pivot(index="VisitorType", columns="Quarter", values="n").reindex(S["levels"]["VisitorType"])[S["levels"]["Quarter"]]
    n = n.fillna(0).astype(int)
    ann = rate.round(1).astype(str) + "%\n(n=" + n.astype(str) + ")"
    ann = ann.where(n > 0, "no sessions")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    sns.heatmap(rate, annot=ann, fmt="", cmap="YlOrRd", vmin=0, vmax=S["rate_ymax"], ax=ax, cbar_kws={"label": "purchase rate (%)"})
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(title, y=1.0, fontsize=12); fig.tight_layout(); save(fig, path)


def monthly_table(df):
    t = rate_table(df, "Month").set_index("Month").reindex(MONTH_ORDER)
    t["share"] = 100 * t.n / len(df)
    t["mean_PageValues"] = df.groupby("Month").PageValues.mean().reindex(MONTH_ORDER)
    return t


def trend_limits(real, syn):
    ts = [monthly_table(d) for d in (real, syn)]
    return {"share": 1.15 * max(t.share.max() for t in ts), "pv": 1.15 * max(t.mean_PageValues.max() for t in ts)}


def fig_monthly(df, S, color, title, path, lim):
    t = monthly_table(df); x = np.arange(len(MONTH_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    axes[0].plot(x, t.share, marker="o", color=color); axes[0].set_ylim(0, lim["share"]); axes[0].set_ylabel("% of all sessions")
    axes[0].set_title("Sessions per month (share of year)", fontsize=9)
    axes[1].errorbar(x, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], marker="o", color=color, capsize=2)
    axes[1].set_ylim(0, S["rate_ymax"] + 10); axes[1].set_ylabel("purchase rate (%)"); axes[1].set_title("Purchase rate per month (95% interval)", fontsize=9)
    axes[2].plot(x, t.mean_PageValues, marker="o", color=color); axes[2].set_ylim(0, lim["pv"]); axes[2].set_ylabel("mean PageValues")
    axes[2].set_title("Mean PageValues per month", fontsize=9)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(MONTH_ORDER, fontsize=7); ax.tick_params(labelsize=7)
    fig.suptitle(title + "\n(only the ten months present in the data; January and April are missing)", y=1.07, fontsize=12)
    fig.tight_layout(); save(fig, path)


# ----------------------------------------------------------------------------------------------------------
# multivariate evidence
# ----------------------------------------------------------------------------------------------------------
def multivariate_evidence(df):
    rows = []
    for v, g in df.groupby("VisitorType"):
        rho = stats.spearmanr(g.ExitRates, g.PageValues)[0]
        rows.append({"pattern": "ExitRates ~ PageValues within visitor type (Spearman rho)", "group": v, "value": rho, "n": len(g)})
    for v, g in df.groupby("Revenue"):
        rho = stats.spearmanr(g.Total_Pages, g.PageValues)[0]
        rows.append({"pattern": "Total_Pages ~ PageValues within outcome (Spearman rho)", "group": v, "value": rho, "n": len(g)})
    t = matrix_table(df)
    for q, g in t.groupby("Quarter"):
        new = g.loc[g.VisitorType == "New_Visitor", "rate"].iloc[0]; ret = g.loc[g.VisitorType == "Returning_Visitor", "rate"].iloc[0]
        rows.append({"pattern": "Purchase-rate gap New minus Returning visitors (pp) by quarter", "group": q, "value": new - ret, "n": int(g.n.sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Scenario 3: combined dataset
# ----------------------------------------------------------------------------------------------------------
def combine(real, syn):
    return pd.concat([real.assign(Data_Source="Real"), syn.assign(Data_Source="Synthetic")], ignore_index=True)


def group_comparison(comb):
    r, s = comb[comb.Data_Source == "Real"], comb[comb.Data_Source == "Synthetic"]
    rows = []
    for c in NUM:
        sd = np.sqrt((r[c].var() + s[c].var()) / 2)
        ks = stats.ks_2samp(r[c], s[c]); mw = stats.mannwhitneyu(r[c], s[c])
        rows.append({"column": c, "mean Real": r[c].mean(), "mean Synthetic": s[c].mean(), "mean Combined": comb[c].mean(),
                     "mean diff (%)": 100 * (s[c].mean() - r[c].mean()) / r[c].mean() if r[c].mean() else np.nan,
                     "std Real": r[c].std(), "std Synthetic": s[c].std(), "std Combined": comb[c].std(),
                     "Cohen d": (s[c].mean() - r[c].mean()) / sd if sd else np.nan, "KS": ks.statistic, "KS p": ks.pvalue,
                     "Mann-Whitney p": mw.pvalue})
    return pd.DataFrame(rows).set_index("column")


def category_proportions(comb):
    rows = []
    for c in CAT:
        ct = pd.crosstab(comb[c], comb.Data_Source, normalize="columns") * 100
        chi2, p, dof, v = cramers_v(comb[c], comb.Data_Source)
        for lev in ct.index:
            rows.append({"column": c, "level": lev, "Real %": ct.loc[lev, "Real"], "Synthetic %": ct.loc[lev, "Synthetic"],
                         "diff (pp)": ct.loc[lev, "Synthetic"] - ct.loc[lev, "Real"], "chi2 p": p, "Cramer's V": v})
    return pd.DataFrame(rows)


def relationship_comparison(comb):
    r, s = comb[comb.Data_Source == "Real"], comb[comb.Data_Source == "Synthetic"]
    pairs = list(SCATTER_PAIRS) + [("Administrative", "Administrative_Duration"), ("Total_Duration", "PageValues"),
                                   ("BounceRates", "Avg_Time_Per_Page")]
    rows = []
    for x, y in pairs:
        vals = {n: d[x].corr(d[y], method="spearman") for n, d in (("Real", r), ("Synthetic", s), ("Combined", comb))}
        rows.append({"pair": f"{x} ~ {y}", "Spearman Real": vals["Real"], "Spearman Synthetic": vals["Synthetic"],
                     "Spearman Combined": vals["Combined"], "Pearson Real": r[x].corr(r[y]), "Pearson Synthetic": s[x].corr(s[y]),
                     "Pearson Combined": comb[x].corr(comb[y])})
    return pd.DataFrame(rows).set_index("pair")


def outlier_comparison(comb):
    r, s = comb[comb.Data_Source == "Real"], comb[comb.Data_Source == "Synthetic"]
    rows = []
    for c in NUM:
        def rate(d, ref):
            q1, q3 = ref[c].quantile(.25), ref[c].quantile(.75); i = q3 - q1
            return 100 * ((d[c] < q1 - 1.5 * i) | (d[c] > q3 + 1.5 * i)).mean()
        lev = stats.levene(r[c], s[c], center="median")
        rows.append({"column": c, "outliers Real (own fences) %": rate(r, r), "outliers Synthetic (own fences) %": rate(s, s),
                     "outliers Combined (combined fences) %": rate(comb, comb),
                     "synthetic rows beyond Real fences %": rate(s, r),
                     "max Real": r[c].max(), "max Synthetic": s[c].max(), "max Combined": comb[c].max(),
                     "std Real": r[c].std(), "std Synthetic": s[c].std(), "std Combined": comb[c].std(),
                     "std Combined / std Real": comb[c].std() / r[c].std() if r[c].std() else np.nan, "Levene p (Real vs Synth)": lev.pvalue})
    return pd.DataFrame(rows).set_index("column")


def fig_combined_hist(comb, S, title, path):
    fig, axes = _grid(len(NUM))
    for ax, c in zip(axes, NUM):
        for src in ("Real", "Synthetic"):
            x = comb.loc[comb.Data_Source == src, c]
            ax.hist(x.clip(upper=S["hi"][c]), bins=S["bins"][c], weights=np.full(len(x), 100 / len(x)), color=SCENARIO_COLOR[src],
                    alpha=.55, label=src, edgecolor="white", linewidth=.2)
        ax.set_ylim(0, S["hist_ymax"][c]); ax.set_xlim(S["bins"][c][0], S["bins"][c][-1])
        ax.set_title(c); ax.set_ylabel("% of sessions in source", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(title + "\n(same bins and axes as Scenarios 1 and 2; each source is expressed as % of its own rows)", y=1.0, fontsize=12)
    fig.tight_layout(); save(fig, path)


def fig_grouped_box(comb, S, title, path):
    fig, axes = _grid(len(NUM))
    for ax, c in zip(axes, NUM):
        sns.boxplot(data=comb, x="Data_Source", y=c, order=["Real", "Synthetic"], hue="Data_Source", palette=SCENARIO_COLOR, ax=ax,
                    showfliers=False, legend=False, width=.55)
        ax.set_ylim(0, S["hi"][c] * 1.05); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(c); ax.tick_params(labelsize=7)
    fig.suptitle(title + "\n(outlier points hidden; y-axis limit = real 99th percentile)", y=1.0, fontsize=12)
    fig.tight_layout(); save(fig, path)


def fig_violins(comb, S, title, path):
    fig, axes = _grid(len(NUM))
    for ax, c in zip(axes, NUM):
        d = comb[["Data_Source", c]].copy(); d[c] = np.log1p(d[c] * (100 if c in ("BounceRates", "ExitRates", "Product_Page_Share", "SpecialDay") else 1))
        sns.violinplot(data=d, x="Data_Source", y=c, order=["Real", "Synthetic"], hue="Data_Source", palette=SCENARIO_COLOR, ax=ax,
                       cut=0, inner="quartile", legend=False, density_norm="width")
        ax.set_xlabel(""); ax.set_ylabel("log(1 + x)", fontsize=8); ax.set_title(c); ax.tick_params(labelsize=7)
    fig.suptitle(title + "\n(log(1 + x) scale; rates and shares multiplied by 100 first)", y=1.0, fontsize=12)
    fig.tight_layout(); save(fig, path)


def fig_grouped_cat_bars(comb, S, title, path):
    fig, axes = _grid(len(CAT))
    for ax, c in zip(axes, CAT):
        t = (100 * pd.crosstab(comb[c], comb.Data_Source, normalize="columns")).reindex(S["levels"][c])
        x = np.arange(len(t))
        for i, src in enumerate(("Real", "Synthetic")):
            ax.bar(x + (i - .5) * .4, t[src], .4, color=SCENARIO_COLOR[src], label=src)
        ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=75 if len(t) > 4 else 20, ha="right", fontsize=7)
        ax.set_ylim(0, S["cat_ymax"][c]); ax.set_title(c); ax.set_ylabel("% within source", fontsize=8); ax.tick_params(axis="y", labelsize=7)
    axes[0].legend(fontsize=7); fig.suptitle(title, y=1.0, fontsize=12); fig.tight_layout(); save(fig, path)


def fig_combined_heatmap(comb, title, path):
    r, s = comb[comb.Data_Source == "Real"], comb[comb.Data_Source == "Synthetic"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    sns.heatmap(comb[NUM].corr(), ax=axes[0], cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"shrink": .7}, linewidths=.3); axes[0].set_title("Combined data: Pearson correlation")
    diff = comb[NUM].corr() - r[NUM].corr()
    sns.heatmap(diff, ax=axes[1], cmap="RdBu_r", vmin=-.5, vmax=.5, center=0, annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"shrink": .7}, linewidths=.3); axes[1].set_title("Combined minus Real (change in Pearson correlation)")
    for ax in axes: ax.tick_params(labelsize=7)
    fig.suptitle(title, y=1.0, fontsize=12); fig.tight_layout(); save(fig, path)


def fig_scatter_by_source(comb, S, title, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (x, y) in zip(axes, SCATTER_PAIRS):
        for src in ("Real", "Synthetic"):
            d = comb[comb.Data_Source == src]
            ax.scatter(d[x], d[y], s=5, alpha=.2, color=SCENARIO_COLOR[src], label=src, rasterized=True)
        ax.set_xlim(0, S["hi"][x] * 1.05); ax.set_ylim(0, S["hi"][y] * 1.05)
        ax.set_xlabel(x, fontsize=8); ax.set_ylabel(y, fontsize=8); ax.set_title(f"{x} vs {y}", fontsize=9); ax.tick_params(labelsize=7)
    axes[0].legend(markerscale=3, fontsize=7)
    fig.suptitle(title, y=1.04, fontsize=12); fig.tight_layout(); save(fig, path)


def fig_overall_comparison(gc, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    order = gc.sort_values("KS").index
    axes[0].barh(order, gc.loc[order, "KS"], color="#7F7F7F"); axes[0].axvline(.05, ls="--", color="k", lw=.8)
    axes[0].set_title("KS statistic (Real vs Synthetic)\ndashed line = 0.05 rule of thumb")
    order = gc["Cohen d"].abs().sort_values().index
    axes[1].barh(order, gc.loc[order, "Cohen d"].abs(), color="#7F7F7F"); axes[1].axvline(.2, ls="--", color="k", lw=.8)
    axes[1].set_title("|Cohen d| of the mean difference\ndashed line = 0.2 (small effect)")
    order = gc["mean diff (%)"].abs().sort_values().index
    axes[2].barh(order, gc.loc[order, "mean diff (%)"], color=["#D1495B" if v < 0 else "#4C78A8" for v in gc.loc[order, "mean diff (%)"]])
    axes[2].axvline(0, color="k", lw=.8); axes[2].set_title("Mean difference, Synthetic vs Real (%)")
    for ax in axes: ax.tick_params(labelsize=8)
    fig.suptitle(title, y=1.02, fontsize=12); fig.tight_layout(); save(fig, path)


def _rank_biserial(df, v):
    a = df.loc[df.Revenue == "Purchase", v]; b = df.loc[df.Revenue == "No_Purchase", v]
    return 2 * stats.mannwhitneyu(a, b)[0] / (len(a) * len(b)) - 1


def bias_table(comb):
    """Does combining change the picture? Same measures in Real, Synthetic and Combined."""
    parts = {"Real": comb[comb.Data_Source == "Real"], "Synthetic": comb[comb.Data_Source == "Synthetic"], "Combined": comb}
    rows = {}
    for name, d in parts.items():
        k, n = int((d.Revenue == "Purchase").sum()), len(d); lo, hi = wilson(k, n)
        col = {"sessions": n, "purchase rate (%)": 100 * k / n, "purchase rate, 95% lower": 100 * lo, "purchase rate, 95% upper": 100 * hi}
        for v in ["PageValues", "ExitRates", "Total_Pages", "Total_Duration"]:
            col[f"rank-biserial: {v} by Revenue"] = _rank_biserial(d, v)
        for c in ["VisitorType", "Quarter", "Month", "Session_Length_Band", "Weekend"]:
            col[f"Cramer's V: {c} x Revenue"] = cramers_v(d[c], d.Revenue)[3]
        rows[name] = col
    return pd.DataFrame(rows)


def fig_outlier_rates(oc, title, path):
    fig, ax = plt.subplots(figsize=(15, 4.6)); x = np.arange(len(oc)); w = .2
    series = [("outliers Real (own fences) %", "Real, own fences", SCENARIO_COLOR["Real"], None),
              ("outliers Synthetic (own fences) %", "Synthetic, own fences", SCENARIO_COLOR["Synthetic"], None),
              ("outliers Combined (combined fences) %", "Combined, combined fences", SCENARIO_COLOR["Combined"], None),
              ("synthetic rows beyond Real fences %", "Synthetic rows outside the real fences", SCENARIO_COLOR["Synthetic"], "//")]
    for i, (c, lab, col, h) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, oc[c], w, color=col, hatch=h, edgecolor="white" if h is None else "black", label=lab, linewidth=.4)
    ax.set_xticks(x); ax.set_xticklabels(oc.index, rotation=35, ha="right", fontsize=8); ax.set_ylabel("% of rows flagged (1.5 x IQR)")
    ax.legend(fontsize=8); fig.suptitle(title, y=1.0, fontsize=12); fig.tight_layout(); save(fig, path)


def fig_bias(bt, title, path):
    cols = ["Real", "Synthetic", "Combined"]; colors = [SCENARIO_COLOR[c] for c in cols]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]; rate = bt.loc["purchase rate (%)", cols].values.astype(float)
    lo, hi = bt.loc["purchase rate, 95% lower", cols].values.astype(float), bt.loc["purchase rate, 95% upper", cols].values.astype(float)
    ax.bar(cols, rate, color=colors, yerr=[rate - lo, hi - rate], capsize=4); ax.set_ylim(0, 20); ax.set_ylabel("purchase rate (%)")
    ax.set_title("Purchase rate (95% Wilson interval)", fontsize=10)
    for ax, pref, ttl in ((axes[1], "rank-biserial: ", "Effect of purchase on numerical variables\n(rank-biserial correlation)"),
                          (axes[2], "Cramer's V: ", "Association of categorical variables with purchase\n(Cramér's V)")):
        rows = [r for r in bt.index if r.startswith(pref)]; x = np.arange(len(rows))
        for i, c in enumerate(cols):
            ax.bar(x + (i - 1) * .27, bt.loc[rows, c].astype(float), .27, color=colors[i], label=c)
        ax.set_xticks(x); ax.set_xticklabels([r.replace(pref, "").replace(" by Revenue", "").replace(" x Revenue", "") for r in rows], rotation=25, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=10)
    axes[1].legend(fontsize=8); fig.suptitle(title, y=1.04, fontsize=12); fig.tight_layout(); save(fig, path)
