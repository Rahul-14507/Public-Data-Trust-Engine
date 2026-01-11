"""
Aadhaar Data Quality Dashboard
==============================
Streamlit dashboard for analyzing Aadhaar data quality.
Supports direct file uploads for cloud deployment.

Run with: streamlit run dashboard.py
"""

import sys
import json
import io
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats


# Page configuration
st.set_page_config(
    page_title="Aadhaar Data Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size: 48px !important; font-weight: bold; }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA QUALITY ENGINE (Embedded for Cloud Deployment)
# ============================================================================

@dataclass
class QualityCheckResult:
    """Container for quality check results."""
    check_name: str
    passed: bool
    severity: int
    details: Dict[str, Any] = field(default_factory=dict)


class DataNormalizer:
    """Normalizes Aadhaar datasets."""
    
    PINCODE_PATTERN = re.compile(r'^\d{6}$')
    
    def __init__(self):
        self.stats = {}
    
    def normalize_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        df = df.copy()
        
        if 'date' in df.columns:
            df['date_normalized'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            failed = df['date_normalized'].isna().sum()
            df['date'] = df['date_normalized'].dt.strftime('%Y-%m-%d')
            df.drop('date_normalized', axis=1, inplace=True)
            self.stats[f'{dataset_name}_date_error_pct'] = (failed / len(df)) * 100
        
        if 'pincode' in df.columns:
            df['pincode_str'] = df['pincode'].astype(str)
            df['pincode_valid'] = df['pincode_str'].str.match(self.PINCODE_PATTERN, na=False)
            invalid = (~df['pincode_valid']).sum()
            df.drop('pincode_str', axis=1, inplace=True)
            self.stats[f'{dataset_name}_pincode_error_pct'] = (invalid / len(df)) * 100
        
        return df
    
    def get_stats(self):
        return self.stats


class QualityChecker:
    """Performs data quality checks."""
    
    def __init__(self):
        self.results = []
        self.anomalies = []
        self.metrics = {}
    
    def run_all_checks(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[List, pd.DataFrame]:
        for name, df in datasets.items():
            self._check_broken_fields(df, name)
            self._check_duplicates(df, name)
            self._check_outliers(df, name)
            if 'date' in df.columns:
                self._check_temporal_integrity(df, name)
        
        if all(k in datasets for k in ['enrolment', 'demographic', 'biometric']):
            self._check_cross_dataset(datasets)
        
        return self.results, pd.DataFrame(self.anomalies) if self.anomalies else pd.DataFrame()
    
    def _check_broken_fields(self, df: pd.DataFrame, name: str):
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]
        broken = []
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                broken.append({'column': col, 'reason': 'All null', 'severity': 10})
                continue
            
            zero_pct = (data == 0).sum() / len(df) * 100
            variance = data.var() if len(data) > 1 else 0
            
            if zero_pct > 90:
                broken.append({'column': col, 'reason': f'Zero: {zero_pct:.1f}%', 'severity': 8})
            elif variance < 0.001 and len(data) > 100:
                broken.append({'column': col, 'reason': f'Low variance: {variance:.6f}', 'severity': 7})
        
        self.results.append(QualityCheckResult('broken_fields', len(broken) == 0, 
            max([b['severity'] for b in broken]) if broken else 0, 
            {'dataset': name, 'broken_fields': broken}))
        self.metrics[f'{name}_broken_fields'] = len(broken)
    
    def _check_duplicates(self, df: pd.DataFrame, name: str):
        check_cols = [c for c in df.columns if not c.startswith('_')]
        exact_dups = df.duplicated(subset=check_cols, keep=False).sum()
        dup_pct = (exact_dups / len(df)) * 100
        
        key_cols = [c for c in ['date', 'pincode', 'district'] if c in df.columns]
        near_dups = df.duplicated(subset=key_cols, keep=False).sum() if key_cols else 0
        near_pct = (near_dups / len(df)) * 100 if key_cols else 0
        
        for idx in df[df.duplicated(subset=check_cols, keep=False)].index[:50]:
            self.anomalies.append({'dataset': name, 'row_index': idx, 'anomaly_type': 'duplicate', 'severity': 5, 'details': 'Duplicate row'})
        
        self.results.append(QualityCheckResult('duplicates', exact_dups == 0,
            min(10, int(dup_pct / 5)), {'dataset': name, 'exact_duplicates': int(exact_dups), 
            'exact_duplicate_pct': dup_pct, 'near_duplicates': int(near_dups)}))
        self.metrics[f'{name}_duplicate_pct'] = dup_pct
        self.metrics[f'{name}_near_duplicate_pct'] = near_pct
    
    def _check_outliers(self, df: pd.DataFrame, name: str):
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not c.startswith('_')]
        total_outliers = 0
        
        for col in numeric_cols[:5]:  # Limit for performance
            data = df[col].dropna()
            if len(data) < 10:
                continue
            
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR))
            
            for idx in df[outliers].index[:20]:
                self.anomalies.append({'dataset': name, 'row_index': idx, 'anomaly_type': 'outlier', 
                    'severity': 4, 'details': f'Outlier in {col}'})
                total_outliers += 1
        
        pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
        self.results.append(QualityCheckResult('outliers', pct < 5, min(10, int(pct)), 
            {'dataset': name, 'total_outliers': total_outliers, 'outlier_pct': pct}))
        self.metrics[f'{name}_outlier_pct'] = pct
    
    def _check_temporal_integrity(self, df: pd.DataFrame, name: str):
        dates = pd.to_datetime(df['date'], errors='coerce').dropna()
        if len(dates) == 0:
            return
        
        min_d, max_d = dates.min(), dates.max()
        expected = pd.date_range(start=min_d, end=max_d, freq='D')
        missing = len(expected) - len(dates.dt.date.unique())
        
        self.results.append(QualityCheckResult('temporal_integrity', missing == 0,
            min(10, missing // 10), {'dataset': name, 'date_range': f'{min_d.date()} to {max_d.date()}',
            'missing_dates': missing, 'sudden_changes': []}))
        self.metrics[f'{name}_temporal_gaps'] = missing
    
    def _check_cross_dataset(self, datasets):
        abnormal = 0
        comparisons = []
        
        for ds_pair in [('enrolment', 'demographic'), ('enrolment', 'biometric')]:
            ds1, ds2 = ds_pair
            if ds1 not in datasets or ds2 not in datasets:
                continue
            
            # Simple size comparison
            ratio = len(datasets[ds2]) / len(datasets[ds1]) if len(datasets[ds1]) > 0 else 0
            is_abnormal = ratio < 0.1 or ratio > 10
            if is_abnormal:
                abnormal += 1
            comparisons.append({'comparison': f'{ds1}_vs_{ds2}', 'abnormal_count': 1 if is_abnormal else 0, 'abnormal_pct': 100 if is_abnormal else 0})
        
        self.results.append(QualityCheckResult('cross_dataset', abnormal == 0, abnormal * 3, {'comparisons': comparisons}))
        self.metrics['cross_dataset_abnormal'] = abnormal
    
    def get_metrics(self):
        return self.metrics


class ReliabilityScorer:
    """Calculates reliability score."""
    
    def __init__(self):
        self.score = 100.0
        self.breakdown = {'base_score': 100, 'penalties': {}, 'total_penalty': 0, 'final_score': 100}
    
    def calculate_score(self, metrics: Dict, normalizer_stats: Dict) -> float:
        penalties = {}
        
        # Duplicates
        dup_pcts = [v for k, v in metrics.items() if 'duplicate_pct' in k and 'near' not in k]
        penalties['duplicate'] = min(20, sum(dup_pcts) / len(dup_pcts) * 2 if dup_pcts else 0)
        
        # Broken fields
        broken = sum(v for k, v in metrics.items() if 'broken_fields' in k)
        penalties['broken_fields'] = min(25, broken * 5)
        
        # Outliers
        outlier_pcts = [v for k, v in metrics.items() if 'outlier_pct' in k]
        penalties['outlier'] = min(15, sum(outlier_pcts) / len(outlier_pcts) * 1.5 if outlier_pcts else 0)
        
        # Date errors
        date_errs = [v for k, v in normalizer_stats.items() if 'date_error' in k]
        penalties['date_error'] = min(15, sum(date_errs) / len(date_errs) * 2 if date_errs else 0)
        
        # Temporal gaps
        gaps = sum(v for k, v in metrics.items() if 'temporal_gaps' in k)
        penalties['temporal_gap'] = min(10, gaps * 0.5)
        
        # Cross-dataset
        cross = metrics.get('cross_dataset_abnormal', 0)
        penalties['cross_dataset'] = min(10, cross * 3)
        
        total = sum(penalties.values())
        self.score = max(0, 100 - total)
        self.breakdown = {'base_score': 100, 'penalties': penalties, 'total_penalty': total, 'final_score': self.score}
        
        return self.score
    
    def get_grade(self):
        if self.score >= 90: return 'A'
        elif self.score >= 80: return 'B'
        elif self.score >= 70: return 'C'
        elif self.score >= 60: return 'D'
        return 'F'
    
    def get_health_status(self):
        if self.score >= 90: return "Excellent - Data is highly reliable"
        elif self.score >= 80: return "Good - Minor quality issues"
        elif self.score >= 70: return "Fair - Some concerns require attention"
        elif self.score >= 60: return "Poor - Significant issues present"
        return "Critical - Data reliability severely compromised"
    
    def get_breakdown(self):
        return self.breakdown


# ============================================================================
# DASHBOARD UI
# ============================================================================

def get_score_emoji(score):
    if score >= 80: return "✅"
    elif score >= 70: return "⚠️"
    elif score >= 60: return "🟡"
    return "❌"


def analyze_files(uploaded_files):
    """Analyze uploaded CSV files."""
    datasets = {}
    row_counts = {}
    
    for f in uploaded_files:
        name_lower = f.name.lower()
        if 'enrol' in name_lower:
            ds_name = 'enrolment'
        elif 'demo' in name_lower:
            ds_name = 'demographic'
        elif 'bio' in name_lower:
            ds_name = 'biometric'
        else:
            ds_name = Path(f.name).stem
        
        df = pd.read_csv(f)
        df['_source'] = f.name
        
        if ds_name in datasets:
            datasets[ds_name] = pd.concat([datasets[ds_name], df], ignore_index=True)
        else:
            datasets[ds_name] = df
        
        if ds_name not in row_counts:
            row_counts[ds_name] = {'files': {}, 'total': 0}
        row_counts[ds_name]['files'][f.name] = len(df)
        row_counts[ds_name]['total'] = len(datasets[ds_name])
    
    # Normalize
    normalizer = DataNormalizer()
    for name, df in datasets.items():
        datasets[name] = normalizer.normalize_dataset(df, name)
    
    # Check quality
    checker = QualityChecker()
    results, anomalies_df = checker.run_all_checks(datasets)
    
    # Score
    scorer = ReliabilityScorer()
    score = scorer.calculate_score(checker.get_metrics(), normalizer.get_stats())
    
    return {
        'datasets': datasets,
        'row_counts': row_counts,
        'quality_results': results,
        'anomalies_df': anomalies_df,
        'reliability_score': score,
        'grade': scorer.get_grade(),
        'health_status': scorer.get_health_status(),
        'score_breakdown': scorer.get_breakdown(),
        'metrics': checker.get_metrics()
    }


def display_results(r):
    """Display analysis results."""
    score = r['reliability_score']
    emoji = get_score_emoji(score)
    
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health Score", f"{emoji} {score:.1f}/100")
    c2.metric("Grade", r['grade'])
    anomaly_count = len(r['anomalies_df']) if not r['anomalies_df'].empty else 0
    c3.metric("Anomalies", f"{anomaly_count:,}")
    total_rows = sum(rc['total'] for rc in r['row_counts'].values())
    c4.metric("Total Rows", f"{total_rows:,}")
    
    st.markdown("---")
    
    # Status
    status = r['health_status']
    if score >= 80: st.success(f"🟢 {status}")
    elif score >= 60: st.warning(f"🟡 {status}")
    else: st.error(f"🔴 {status}")
    
    # Tabs
    t1, t2, t3, t4 = st.tabs(["📈 Score", "📊 Data", "🔍 Anomalies", "⚠️ Issues"])
    
    with t1:
        st.subheader("Score Breakdown")
        penalties = r['score_breakdown'].get('penalties', {})
        if penalties:
            pdf = pd.DataFrame([{'Category': k.replace('_', ' ').title(), 'Penalty': v} for k, v in penalties.items() if v > 0])
            if not pdf.empty:
                st.bar_chart(pdf.set_index('Category'))
            for cat, pen in penalties.items():
                st.markdown(f"**{cat.replace('_', ' ').title()}:** -{pen:.2f} pts")
    
    with t2:
        st.subheader("Datasets")
        for name, counts in r['row_counts'].items():
            with st.expander(f"📁 {name.capitalize()} ({counts['total']:,} rows)", expanded=True):
                for fname, rows in counts['files'].items():
                    st.markdown(f"- {fname}: {rows:,} rows")
                if name in r['datasets']:
                    cols = [c for c in r['datasets'][name].columns if not c.startswith('_')]
                    st.dataframe(r['datasets'][name][cols].head(5), use_container_width=True)
    
    with t3:
        adf = r['anomalies_df']
        if not adf.empty:
            c1, c2 = st.columns(2)
            with c1:
                if 'anomaly_type' in adf.columns:
                    st.bar_chart(adf['anomaly_type'].value_counts())
            with c2:
                if 'dataset' in adf.columns:
                    st.bar_chart(adf['dataset'].value_counts())
            st.dataframe(adf.head(100), use_container_width=True)
        else:
            st.success("🎉 No anomalies!")
    
    with t4:
        for res in r['quality_results']:
            icon = "✅" if res.passed else "❌"
            with st.expander(f"{icon} {res.check_name.replace('_', ' ').title()}"):
                st.markdown(f"**Severity:** {res.severity}/10")
                if res.check_name == 'broken_fields':
                    for bf in res.details.get('broken_fields', []):
                        st.markdown(f"- `{bf['column']}`: {bf['reason']}")
                elif res.check_name == 'duplicates':
                    st.markdown(f"- Exact: {res.details.get('exact_duplicates', 0):,} ({res.details.get('exact_duplicate_pct', 0):.1f}%)")
                elif res.check_name == 'temporal_integrity':
                    st.markdown(f"- Range: {res.details.get('date_range', 'N/A')}")
                    st.markdown(f"- Missing dates: {res.details.get('missing_dates', 0)}")


def main():
    st.title("📊 Aadhaar Data Quality Intelligence Engine")
    st.markdown("Upload your CSV files for instant data quality analysis.")
    st.markdown("---")
    
    # File upload
    uploaded = st.file_uploader(
        "📤 Upload CSV files (Enrolment, Demographic, Biometric)",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s):** " + ", ".join(f.name for f in uploaded))
        
        if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    st.session_state['results'] = analyze_files(uploaded)
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if 'results' in st.session_state:
        st.markdown("---")
        display_results(st.session_state['results'])
    
    st.markdown("---")
    st.caption("Aadhaar Data Quality Intelligence Engine | Built with Streamlit")


if __name__ == '__main__':
    main()
