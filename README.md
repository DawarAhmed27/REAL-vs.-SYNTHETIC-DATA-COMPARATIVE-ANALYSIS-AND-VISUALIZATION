# Real vs. Synthetic Data: Comparative Analysis and Visualization

Final lab project for **Data Analysis and Visualization** (BS Data Science, FAST-NUCES Islamabad, Fall 2026).

A real-world dataset is cleaned, a synthetic version is generated from it, and the real, synthetic and combined datasets are analysed and compared in Python and Power BI to judge how faithfully the synthetic data represents the real data.

> Work in progress. Sections marked _TBD_ are filled in as each phase is completed.

## Dataset
- Name / domain: _TBD (Phase 1)_
- Source and link: _TBD_
- Records / attributes: _TBD_

## Repository structure
```
data/            original (never modified), cleaned, synthetic, combined CSVs
notebooks/       Jupyter notebooks, run in numeric order
figures/         Python figures: real/, synthetic/, combined/, comparative/
powerbi/         .pbix report, theme and PDF export
report/          Final report source
docs/            Project plan and notes
```

## Environment
- Python: _TBD_  |  Power BI Desktop: _TBD_
- Libraries: see `requirements.txt` (pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, SDV)
- Fixed random seed: _TBD_

## How to run
1. Create a virtual environment and run `pip install -r requirements.txt`.
2. Place the original dataset in `data/original/` (see the dataset section for the download link).
3. Run the notebooks in `notebooks/` in numeric order.
4. Open `powerbi/*.pbix` in Power BI Desktop and refresh the data sources to point at `data/`.

## Team
_TBD_
