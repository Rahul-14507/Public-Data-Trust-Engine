# Public Data Trust Engine

A comprehensive data quality pipeline for analyzing Aadhaar datasets (Enrolment, Demographic, Biometric).

## 📊 Features

- **Auto-detect broken or suspicious fields** - Identifies columns with high zero rates, near-zero variance
- **Flag statistical outliers** - Uses IQR and Z-score methods per pincode/day
- **Detect duplicates** - Exact and near-duplicate detection
- **Generate reliability score** - 0-100 score with detailed penalty breakdown
- **Produce automated reports** - JSON, CSV, and Markdown outputs
- **Interactive dashboard** - Streamlit-based visualization

## 📁 Project Structure

```
data_quality_engine/
├── src/
│   ├── __init__.py           # Package exports
│   ├── data_loader.py        # CSV ingestion & logging
│   ├── normalizer.py         # Date/pincode standardization
│   ├── quality_checks.py     # All DQ checks (A-E)
│   ├── scoring.py            # Reliability scoring (0-100)
│   └── reporter.py           # Output generation
├── outputs/                   # Generated reports
│   ├── anomalies_summary.json
│   ├── anomalies_table.csv
│   └── data_quality_report.md
├── run_pipeline.py           # CLI entrypoint
├── dashboard.py              # Streamlit dashboard
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd c:\data_quality_engine
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python run_pipeline.py --input-dir "c:\"
```

### 3. View Reports

Reports are saved to `outputs/`:

- `anomalies_summary.json` - Machine-readable summary
- `anomalies_table.csv` - Row-level anomaly flags
- `data_quality_report.md` - Human-readable report

### 4. Launch Dashboard (Optional)

```bash
streamlit run dashboard.py
```

## 📋 Quality Checks

| Check                     | Description                                         |
| ------------------------- | --------------------------------------------------- |
| **A) Broken Fields**      | Detects columns with >90% zeros, near-zero variance |
| **B) Duplicates**         | Exact and near-duplicate detection                  |
| **C) Outliers**           | IQR + Z-score methods                               |
| **D) Temporal Integrity** | Missing dates, sudden volume changes                |
| **E) Cross-Dataset**      | Enrolment vs Update ratio validation                |

## 📈 Reliability Score

Score Formula (start at 100):

- Duplicate penalty: -(pct × 2), max 20 pts
- Broken fields: -(count × 5), max 25 pts
- Outliers: -(pct × 1.5), max 15 pts
- Date errors: -(pct × 2), max 15 pts
- Missing data: -(pct × 1), max 15 pts
- Temporal gaps: -(days × 0.5), max 10 pts

## 📝 CLI Options

```bash
python run_pipeline.py --help

Options:
  --input-dir   Directory containing dataset folders
  --output-dir  Directory for output reports (default: ./outputs)
  -v, --verbose Enable debug logging
```

## 📦 Exit Codes

- `0` - Data quality acceptable (score ≥ 70)
- `1` - Data quality concerns (score 50-69)
- `2` - Critical issues (score < 50)
