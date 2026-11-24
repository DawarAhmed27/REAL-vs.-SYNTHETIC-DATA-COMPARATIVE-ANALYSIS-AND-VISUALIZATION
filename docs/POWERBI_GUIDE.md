# Power BI build guide (Phase 8)

Everything below uses files that already exist in the repository. You build the `.pbix` yourself in Power BI Desktop (Windows). The DAX in `powerbi/dax_measures.dax` was checked against the Python results; the click-by-click steps were written without running Power BI, so if a menu is named differently in your version, tell me and I will correct this guide.

Files in `powerbi/`: `theme_real_vs_synthetic.json`, `dax_measures.dax`, `dax_calculated_columns.dax`, `dim_metric.csv`, `python_parity_targets.csv`.

---

## 0. What the finished report contains

| # | Page name | Purpose | Brief requirement |
|---|-----------|---------|-------------------|
| 1 | `Real` | Scenario 1 dashboard | one page per scenario, KPI cards, slicers, same layout |
| 2 | `Synthetic` | Scenario 2 dashboard (copy of page 1) | same |
| 3 | `Combined` | Scenario 3 dashboard (copy of page 1 plus a `Data_Source` slicer) | same, `Data_Source` as slicer field |
| 4 | `Comparison` | Distributions, mean / median / std bars, box-style chart, categories, monthly trend, scenario slicer | Section 6 comparison charts |
| 5 | `Comparison 2` | Python correlation matrices (images), scatter with fitted lines, "which column differs most" | Section 6 comparison charts |
| 6 | `Python Figures` | KDE, violin, heatmap, PCA images made in Python | Python-only chart types inserted as images |
| 7 | `Data Check` | DAX numbers vs Python numbers, row counts | "numbers must match Python" |
| 8 | `Tooltip` (hidden) | Tooltip page for the month charts | "tooltips" |

Rubric for this part: 10 marks (design, accuracy, interactivity). At least 6 DAX measures are required; the file has about 45, grouped A to G.

## 1. Before you start

1. Install the current Power BI Desktop. Note its version (Help > About) for the README.
2. Turn off automatic relationship detection: File > Options and settings > Options > Current File > Data Load > untick **Autodetect new relationships after data is loaded**. (Otherwise it creates wrong links between the `*_long` tables.)
3. Folder layout (the repo, which is your local `Data-Visualisation-LabProject` folder):

```
data/cleaned/online_shoppers_cleaned.csv        -> table Real
data/synthetic/synthetic_online_shoppers.csv    -> table Synthetic
data/combined/combined_dataset.csv              -> table Combined
powerbi/dim_metric.csv                          -> table dim_metric
results/comparison/powerbi_tables/*.csv         -> dim_scenario, stats_long, box_stats_long,
                                                   histogram_bins_long, categorical_shares_long, monthly_long
```
Do NOT load `rows_by_scenario.csv` (10 MB, not needed) or `correlation_long.csv` (the correlation matrices are Python images).

## 2. Load the data (Home > Get data > Text/CSV)

For each file: choose it, click **Transform Data** (not Load), so you can check types in Power Query.

Rules (the brief says no silent changes, so write each change down in the README):
- Keep **Promoted Headers** and **Changed Type** only. Do not filter, replace or add steps.
- Use **Decimal number** (never "Fixed decimal number": it rounds to 4 decimals and your means will no longer match Python).
- Whole number: `Administrative`, `Informational`, `ProductRelated`, `Total_Pages`.
- Decimal number: the other 10 numeric columns.
- Text: `Month`, `Quarter`, `VisitorType`, `Weekend`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `Session_Length_Band`, `Revenue`, and `Data_Source` (in `Combined`).
- Rename each query exactly as in the table in Section 1 (right-click the query > Rename). The DAX uses these names.
- For the `*_long` tables: `bin_index`, `scenario_order` whole number; the other numeric columns decimal; `column`, `scenario`, `bin_label`, `level`, `Month` text.

Click **Close & Apply**.

## 3. Model

### 3.1 Relationships (Model view > Manage relationships > New). All are one-to-many, single direction.

| From (one side) | To (many side) |
|---|---|
| `dim_scenario[scenario]` | `histogram_bins_long[scenario]` |
| `dim_scenario[scenario]` | `box_stats_long[scenario]` |
| `dim_scenario[scenario]` | `categorical_shares_long[scenario]` |
| `dim_scenario[scenario]` | `monthly_long[scenario]` |
| `dim_metric[metric]` | `histogram_bins_long[column]` |
| `dim_metric[metric]` | `box_stats_long[column]` |

`Real`, `Synthetic`, `Combined` and `stats_long` have no relationships. That is intended: `Combined` holds both sources in one table, and `stats_long` is only a Python reference.

### 3.2 Calculated columns (sort keys only; they do not change any value)
Open `powerbi/dax_calculated_columns.dax` and create each column (Table tools > New column), then set Sort by column as the comments say: `Combined[MonthNo]`, `Combined[BandNo]`, `monthly_long[MonthNo]`, `histogram_bins_long[BinKey]`.

### 3.3 Measures table
Home > Enter data > leave empty, name the table `_Measures`, Load. Then for each block in `powerbi/dax_measures.dax`: select `_Measures` > New measure > paste one measure > Enter. Set the format given in the header comment (Measure tools > Format). Do blocks A and B first; the later blocks depend on them.

### 3.4 Table properties
- `dim_scenario[scenario]`: Sort by column > `scenario_order` (gives Real, Synthetic, Combined).
- `dim_metric[metric]`: Sort by column > `metric_order`.
- Hide `Real`, `Synthetic`, `stats_long` columns from Report view if you like; do not delete the tables.

## 4. Theme
View > Themes > Browse for themes > `powerbi/theme_real_vs_synthetic.json`. If Power BI rejects it, tell me the error message.
Colours: Real blue `#4C78A8`, Synthetic orange `#F58518`, Combined green `#59A14F`, Purchase `#D1495B`, No purchase `#9AA5B1`. These are the same as the Python figures. Check each chart: when the legend is `scenario` or `Data_Source`, set the colours by hand (Format > Visual > Colors) if the automatic order is different.

Fonts: Segoe UI. Page canvas: 16:9 (1280 x 720) for pages 1, 2, 3, 4, 5, 6, 7. Page background `#F4F6F8` comes from the theme.

## 5. Scenario pages (pages 1 to 3)

Build page `Real` first, carefully; pages 2 and 3 are copies.

### 5.1 Layout (Format > Properties > Position and Size; units are pixels)

| ID | Visual | x | y | w | h |
|---|---|---|---|---|---|
| T | Title text box | 16 | 8 | 900 | 36 |
| S1 | Slicer `Combined[Month]` | 16 | 52 | 240 | 40 |
| S2 | Slicer `Combined[VisitorType]` | 266 | 52 | 240 | 40 |
| S3 | Slicer `Combined[Weekend]` | 516 | 52 | 240 | 40 |
| S4 | Slicer `Combined[Session_Length_Band]` | 766 | 52 | 240 | 40 |
| S5 | Slicer `dim_metric[metric]` (single select) | 1016 | 52 | 248 | 40 |
| K1 to K6 | Six KPI cards | 16, 226, 436, 646, 856, 1066 | 98 | 198 | 64 |
| A1 | Histogram (column chart) | 16 | 170 | 416 | 236 |
| A2 | Month combo chart | 440 | 170 | 416 | 236 |
| A3 | Multi-row card: statistics of the selected variable | 864 | 170 | 400 | 236 |
| B1 | Donut: Revenue | 16 | 414 | 208 | 298 |
| B2 | Donut: VisitorType | 232 | 414 | 200 | 298 |
| B3 | Scatter | 440 | 414 | 416 | 298 |
| B4 | Matrix: purchase rate by Month and VisitorType | 864 | 414 | 400 | 298 |

Slicers: Format > Slicer settings > Style > **Dropdown**. Make `Weekend` and `Session_Length_Band` dropdowns too, for equal size.

### 5.2 Content of each visual

**Page filter (Filters pane > Filters on this page):** `Combined[Data_Source]` is `Real`. This is the only difference between the three pages.

**T (title):** "Scenario 1: Real dataset | 12,199 sessions". Pages 2 and 3 change only the text.

**KPI cards** (Card visual, one measure each): `Sessions`, `Purchase Rate`, `Returning Visitor Share`, `Avg Pages per Session`, `Avg Minutes per Session`, `Avg Page Value`. Rename the field label to a clear title (double-click the field in Visualizations).

**A1 Histogram.** Clustered column chart: X-axis `histogram_bins_long[BinKey]`, Y-axis `histogram_bins_long[pct_of_rows]`. Visual-level filter: `histogram_bins_long[scenario]` is `Real` (Synthetic on page 2, Combined on page 3). Title: "Distribution of the selected variable (% of sessions per bin)". Y-axis title "Share of sessions (%)". X-axis title "Bin (value range of the variable chosen in the slicer)". Why precomputed bins: they are the same bins Python used for all three scenarios (shared edges up to the real 99th percentile, with larger values counted in the last bin), so the three pages are directly comparable. Power BI's own binning would choose different widths per page. Edit interactions: this chart does not respond to the Month, VisitorType, Weekend, Session band slicers (precomputed), so add the words "(not filtered by the slicers)" to its subtitle.

**A2 Month combo chart.** Line and clustered column chart. X-axis `Combined[Month]` (calendar order via `MonthNo`); column Y-axis `Sessions`; line Y-axis `Purchase Rate`. Titles: "Sessions (columns) and purchase rate (line) by month". Axis titles: "Month of visit", "Number of sessions", "Purchase rate (%)". This is the trend analysis (Section 5, trends).

**A3 Multi-row card.** Fields: `Mean (selected)`, `Median (selected)`, `Std Dev (selected)`, `Variance (selected)`, `Min (selected)`, `Max (selected)`, `Count (selected)`. Title: "Statistics of the selected variable". These are the required DAX statistics; they respond to S5 and to the page filter.

**B1 Donut.** Legend `Combined[Revenue]`, Values `Sessions`. Colours: Purchase `#D1495B`, No_Purchase `#9AA5B1`. Detail label: category and percent.
**B2 Donut.** Legend `Combined[VisitorType]`, Values `Sessions`.

**B3 Scatter.** X `Combined[ProductRelated]`, Y `Combined[ProductRelated_Duration]`, Legend `Combined[Revenue]`. Format: marker size small, transparency about 60%. Analytics pane > Trend line on. Titles: "Product pages visited vs time on product pages", axis titles "Product pages visited (count)", "Time on product pages (seconds)". Do not add a Details field (it would plot every row as a separate point and slow the page).

**B4 Matrix.** Rows `Combined[Month]`, Columns `Combined[VisitorType]`, Values `Purchase Rate`. Format > Cell elements > Background colour > Gradient (white to `#D1495B`, minimum 0). Title "Purchase rate by month and visitor type".

### 5.3 Make pages 2 and 3
Right-click the `Real` tab > Duplicate page, twice. Rename. On each:
- Page 2: page filter `Data_Source` is `Synthetic`; histogram visual filter `scenario` is `Synthetic`; title text.
- Page 3: remove the `Data_Source` page filter; histogram visual filter `scenario` is `Combined`; add one tile slicer `Combined[Data_Source]` at x 16, y 8 w 330 h 36 (and shrink the title width to 560, x 354); set scatter Legend to `Data_Source` (so the colours show Real vs Synthetic). Everything else is the same.

Interactions to check on every page: click a slice in the donut and watch the other visuals respond (Format > Edit interactions if one should not).

## 6. Comparison page (page 4)

Slicers (all at y 52, h 40): `dim_scenario[scenario]` as **Tile** slicer, multi-select allowed (Ctrl+click), x 16 w 400; `dim_metric[metric]` dropdown single select, x 424 w 300; `categorical_shares_long[column]` dropdown single select, x 732 w 260. Set the metric to `Total_Pages` and the category column to `Month` as defaults.

| ID | Visual | x | y | w | h |
|---|---|---|---|---|---|
| C1 | Three-way histogram | 16 | 100 | 620 | 250 |
| C2 | Mean bar | 644 | 100 | 200 | 250 |
| C3 | Median bar | 852 | 100 | 200 | 250 |
| C4 | Std dev bar | 1060 | 100 | 204 | 250 |
| C5 | Statistics table by scenario | 16 | 358 | 1248 | 130 |
| C6 | Box-style chart | 16 | 496 | 400 | 216 |
| C7 | Categorical comparison | 424 | 496 | 420 | 216 |
| C8 | Monthly purchase-rate trend | 852 | 496 | 412 | 216 |

- **C1:** clustered column. X `histogram_bins_long[BinKey]`, Legend `dim_scenario[scenario]`, Y `histogram_bins_long[pct_of_rows]`. Colours as in Section 4. Title "Distribution of the selected variable: Real vs Synthetic vs Combined".
- **C2, C3, C4:** clustered column. X `dim_scenario[scenario]`, Y `Mean by Scenario`, `Median by Scenario`, `Std Dev by Scenario`. Turn on data labels. Titles: "Mean of the selected variable" and so on. Set each bar colour by scenario.
- **C5:** Table. Columns: `dim_scenario[scenario]`, `Rows by Scenario`, `Mean by Scenario`, `Median by Scenario`, `Std Dev by Scenario`, `Variance by Scenario`, `Min by Scenario`, `Max by Scenario`. This is the comparison table for the chosen variable (Real, Synthetic, Combined).
- **C6 (box-style).** Power BI has no built-in box plot, so build it as a stacked column chart: X `dim_scenario[scenario]`; Y `Box Q1`, `Box Q1 to Median`, `Box Median to Q3` (that order). Format > Columns: colour for `Box Q1` set to **no fill** (transparent) so only the two boxes remain. Tooltips: add `Min by Scenario`, `Max by Scenario`, `Box Upper Fence`, `Box Outlier Share`. Title "Quartile boxes of the selected variable (box from Q1 to Q3, split at the median)". Say in the report that this is a box-style chart without drawn whiskers; the exact whiskers and outliers are in the Python box plots. (Optional alternative: install "Box and Whisker chart" from AppSource and feed it raw `Combined` rows; skip if you cannot rely on internet at the demo.)
- **C7:** clustered bar. Axis `categorical_shares_long[level]`, Legend `dim_scenario[scenario]`, Values `categorical_shares_long[share_pct]`. Title "Category shares (% of sessions) for the chosen column".
- **C8:** line chart. X `monthly_long[Month]` (sorted by `MonthNo`), Y `monthly_long[rate]`, Legend `dim_scenario[scenario]`. Title "Monthly purchase rate (%) by scenario". Tooltip fields `monthly_long[n]`, `lo`, `hi` (95% interval). Note February has only 181 real sessions.

## 7. Comparison 2 page (page 5)

Slicer: `dim_scenario[scenario]` tile (x 16, y 52, w 400, h 40).

| ID | Visual | x | y | w | h |
|---|---|---|---|---|---|
| P1 | Image: `figures/comparison/c04_correlation_pearson_side_by_side.png` | 16 | 100 | 620 | 240 |
| P2 | Image: `figures/comparison/c05_correlation_spearman_side_by_side.png` | 644 | 100 | 620 | 240 |
| P3 | Scatter with fitted lines | 16 | 348 | 420 | 364 |
| P4 | Bar: `Mean % Difference (Synthetic vs Real)` by variable | 444 | 348 | 400 | 364 |
| P5 | Bar: `Std Dev Ratio (Synthetic / Real)` by variable | 852 | 348 | 412 | 364 |

- **P1, P2:** Insert > Image. Add alt text and a text box title each ("Pearson correlation: Real, Synthetic, absolute difference"). They are Python figures; say so in a caption.
- **P3:** scatter from `Combined`: X `ProductRelated`, Y `ProductRelated_Duration`, Legend `Data_Source`; Analytics > Trend line on (one line per source). Subtitle: "Real and Synthetic; the Combined line is in the Python figure c07".
- **P4:** clustered bar. Y-axis `stats_long[column]`, X `Mean % Difference (Synthetic vs Real)`. Sort descending by value. This answers "which numerical column differs most". `stats_long` has no relationship to the slicers, so P4 and P5 always compare Real and Synthetic whatever is selected; say so in the subtitle.
- **P5:** same for `Std Dev Ratio (Synthetic / Real)`.

## 8. Python Figures page (page 6)
Insert > Image for each, with a caption text box and alt text. Size 410 x 215.

| Position (x, y) | Image | Caption |
|---|---|---|
| 16, 60 | `figures/real/scenario1/s1_02_kde.png` | KDE, Real (Python) |
| 435, 60 | `figures/synthetic/scenario2/s2_02_kde.png` | KDE, Synthetic (Python) |
| 854, 60 | `figures/combined/scenario3/s3_03_violins.png` | Violin plots, Combined (Python) |
| 16, 300 | `figures/real/scenario1/s1_09_correlation_heatmaps.png` | Correlation heatmaps, Real (Python) |
| 435, 300 | `figures/synthetic/scenario2/s2_09_correlation_heatmaps.png` | Correlation heatmaps, Synthetic (Python) |
| 854, 300 | `figures/combined/scenario3/s3_06_combined_heatmap.png` | Combined heatmap (Python) |
| 16, 540 | `figures/advanced/a02_pca_scatter.png` (w 510, h 170) | PCA: real and synthetic projected on the real axes (Python) |

## 9. Tooltip page (page 8)
New page > Format page > Page information > **Allow use as tooltip** on; Canvas settings > Type > Tooltip. Put on it: card `Sessions`, card `Purchase Rate`, card `Avg Page Value`. Then in the month charts (A2, B4, C8): Format > General > Tooltips > Type > Report page > choose this page, with `Combined[Month]` as the tooltip field. Hide the page (right-click tab > Hide page).

## 10. Data Check page (page 7)
- Cards: `Rows Real table`, `Rows Synthetic table`, `Rows Combined table`, `Row Check` (must say OK).
- Matrix: Rows `dim_metric[metric]`, Columns `dim_scenario[scenario]`, Values `Mean Difference (DAX - Python)` and `Std Dev Difference (DAX - Python)`. Every cell must be 0 (shown at 6 decimals). Take a screenshot for the report appendix.
- Why they match: `STDEV.S` and `PERCENTILE.INC` are the same formulas as pandas `std()` (n - 1) and `quantile()` (linear interpolation).

## 11. Numbers to check against (from Python; also in `powerbi/python_parity_targets.csv`)

| Measure | Real | Synthetic | Combined |
|---|---|---|---|
| Sessions | 12,199 | 12,199 | 24,398 |
| Purchase Rate | 15.64% | 14.19% | 14.92% |
| Returning Visitor Share | 85.46% | 86.16% | 85.81% |
| Avg Pages per Session | 34.42 | 35.73 | 35.08 |
| Avg Minutes per Session | 21.59 | 22.40 | 22.00 |
| Avg Page Value | 5.95 | 5.75 | 5.85 |

If any number differs: check the column type (Fixed decimal?), the page filter, and whether a slicer is still selected. Explain any remaining difference in the report, as the brief requires.

## 12. Finish
1. Save as `powerbi/Real_vs_Synthetic.pbix`. If it is over 25 MB or GitHub refuses it, do not commit it; submit it separately and say so in the README.
2. File > Export > Export to PDF (all pages). Save as `powerbi/Real_vs_Synthetic_dashboard.pdf`.
3. For the report: screenshot pages 1 to 5 and 7 (Windows: Snipping Tool) with a one-paragraph interpretation each.
4. Add to the README: Power BI Desktop version, the CSV load order, the sort-key calculated columns, "no other Power Query step".
5. Save a copy of the final DAX (File > Export or from this repo) in the report appendix; the viva asks you to explain each measure.

## 13. Viva: what you should be able to say about the DAX
- `Sessions`: count of rows in the current filter context; the page filter on `Data_Source` decides the scenario.
- `Purchase Rate`: `DIVIDE` of purchases by sessions (DIVIDE avoids division-by-zero errors).
- `Mean/Median/Std Dev/Variance/Min/Max (selected)`: one `SWITCH` over the variable picked in the slicer; `STDEV.S` and `VAR.S` are the sample versions (n - 1), the same as pandas.
- `... by Scenario`: `SWITCH` on the scenario axis; `CALCULATE` applies the Real or Synthetic filter; `REMOVEFILTERS` gives all rows = Combined.
- `Box Q1`, `Box Q1 to Median`, `Box Median to Q3`: stacked differences that draw a box; the first is transparent.
- `Mean Difference (DAX - Python)`: proof that Power BI reproduces the Python table.
- Why `Combined` is used for the three scenario pages: one table, one set of measures, so the three pages cannot use different definitions.

## 14. Known limits (state them in the report)
- The histogram and box-style charts read tables that Python precomputed, to guarantee identical bins and exact quartiles; they do not respond to the Month / Visitor / Weekend slicers.
- Whiskers and outlier dots of the Python box plots are not drawn in the Power BI box-style chart.
- Correlation heatmaps, KDE, violins and the PCA plot are Python images because Power BI cannot draw them natively.
