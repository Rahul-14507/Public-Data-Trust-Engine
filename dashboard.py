"""
Aadhaar Data Quality Dashboard
==============================
Streamlit dashboard for visualizing data quality metrics.

Run with: streamlit run dashboard.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


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
    .big-font {
        font-size: 48px !important;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .score-excellent { color: #10b981; }
    .score-good { color: #3b82f6; }
    .score-fair { color: #f59e0b; }
    .score-poor { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


def load_summary_data():
    """Load the anomalies summary JSON."""
    summary_path = Path(__file__).parent / 'outputs' / 'anomalies_summary.json'
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_anomalies_table():
    """Load the anomalies CSV table."""
    csv_path = Path(__file__).parent / 'outputs' / 'anomalies_table.csv'
    
    if not csv_path.exists():
        return pd.DataFrame()
    
    return pd.read_csv(csv_path)


def get_score_color(score):
    """Get color class based on score."""
    if score >= 80:
        return "score-excellent"
    elif score >= 70:
        return "score-good"
    elif score >= 60:
        return "score-fair"
    else:
        return "score-poor"


def get_score_emoji(score):
    """Get emoji based on score."""
    if score >= 80:
        return "✅"
    elif score >= 70:
        return "⚠️"
    elif score >= 60:
        return "🟡"
    else:
        return "❌"


def main():
    """Main dashboard function."""
    
    # Header
    st.title("📊 Aadhaar Data Quality Intelligence Engine")
    st.markdown("---")
    
    # Load data
    summary = load_summary_data()
    anomalies_df = load_anomalies_table()
    
    if summary is None:
        st.error("⚠️ No data found! Please run the pipeline first:")
        st.code("python run_pipeline.py --input-dir c:\\Projects\\UIDAI")
        st.info("After running the pipeline, refresh this page.")
        return
    
    # Sidebar
    st.sidebar.title("📋 Report Info")
    st.sidebar.markdown(f"**Generated:** {summary.get('generated_at', 'N/A')[:19]}")
    st.sidebar.markdown(f"**Grade:** {summary.get('grade', 'N/A')}")
    
    # Dataset selector
    datasets = list(summary.get('datasets', {}).keys())
    if datasets:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Datasets Loaded")
        for ds in datasets:
            rows = summary['datasets'][ds].get('total_rows', 0)
            st.sidebar.markdown(f"- **{ds.capitalize()}:** {rows:,} rows")
    
    # Main content - Health Score
    col1, col2, col3, col4 = st.columns(4)
    
    score = summary.get('health_score', 0)
    emoji = get_score_emoji(score)
    
    with col1:
        st.metric(
            label="Health Score",
            value=f"{emoji} {score:.1f}/100",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Grade",
            value=summary.get('grade', 'N/A'),
            delta=None
        )
    
    with col3:
        st.metric(
            label="Total Anomalies",
            value=f"{summary.get('total_anomalies', 0):,}",
            delta=None
        )
    
    with col4:
        broken = len(summary.get('broken_fields', []))
        st.metric(
            label="Broken Fields",
            value=broken,
            delta=None
        )
    
    st.markdown("---")
    
    # Status message
    status = summary.get('health_status', '')
    if score >= 80:
        st.success(f"🟢 {status}")
    elif score >= 60:
        st.warning(f"🟡 {status}")
    else:
        st.error(f"🔴 {status}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Score Breakdown", 
        "🔍 Anomalies", 
        "📊 Duplicates", 
        "⚠️ Broken Fields"
    ])
    
    # Tab 1: Score Breakdown
    with tab1:
        st.subheader("Reliability Score Breakdown")
        
        breakdown = summary.get('score_breakdown', {})
        penalties = breakdown.get('penalties', {})
        
        if penalties:
            # Create penalty chart data
            penalty_data = pd.DataFrame([
                {'Category': k.replace('_', ' ').title(), 'Penalty': v}
                for k, v in penalties.items()
                if v > 0
            ])
            
            if not penalty_data.empty:
                st.bar_chart(penalty_data.set_index('Category'))
            else:
                st.info("🎉 No penalties applied - excellent data quality!")
            
            # Penalty table
            st.markdown("### Penalty Details")
            col1, col2 = st.columns(2)
            
            with col1:
                for category, penalty in list(penalties.items())[:4]:
                    st.markdown(f"**{category.replace('_', ' ').title()}:** -{penalty:.2f} pts")
            
            with col2:
                for category, penalty in list(penalties.items())[4:]:
                    st.markdown(f"**{category.replace('_', ' ').title()}:** -{penalty:.2f} pts")
            
            st.markdown(f"### Total Penalty: **-{breakdown.get('total_penalty', 0):.2f}** pts")
    
    # Tab 2: Anomalies
    with tab2:
        st.subheader("Anomaly Details")
        
        if not anomalies_df.empty:
            # Summary by type
            if 'anomaly_type' in anomalies_df.columns:
                type_counts = anomalies_df['anomaly_type'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Anomalies by Type")
                    st.bar_chart(type_counts)
                
                with col2:
                    st.markdown("### Anomalies by Dataset")
                    if 'dataset' in anomalies_df.columns:
                        ds_counts = anomalies_df['dataset'].value_counts()
                        st.bar_chart(ds_counts)
            
            # Anomaly table
            st.markdown("### Anomaly Records")
            st.dataframe(
                anomalies_df.head(100),
                use_container_width=True,
                height=400
            )
            
            st.caption(f"Showing first 100 of {len(anomalies_df):,} anomalies")
        else:
            st.info("🎉 No anomalies detected!")
    
    # Tab 3: Duplicates
    with tab3:
        st.subheader("Duplicate Statistics")
        
        dup_stats = summary.get('duplicate_stats', {})
        
        if dup_stats:
            for dataset, stats in dup_stats.items():
                with st.expander(f"📁 {dataset.capitalize()}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        exact_dups = stats.get('exact_duplicates', 0)
                        exact_dups = int(exact_dups) if isinstance(exact_dups, str) else exact_dups
                        st.metric(
                            "Exact Duplicates",
                            f"{exact_dups:,}"
                        )
                    
                    with col2:
                        st.metric(
                            "Duplicate %",
                            f"{stats.get('exact_duplicate_pct', 0):.2f}%"
                        )
                    
                    with col3:
                        near_dups = stats.get('near_duplicates', 0)
                        near_dups = int(near_dups) if isinstance(near_dups, str) else near_dups
                        st.metric(
                            "Near Duplicates",
                            f"{near_dups:,}"
                        )
        else:
            st.info("No duplicate statistics available.")
    
    # Tab 4: Broken Fields
    with tab4:
        st.subheader("Broken Field Alerts")
        
        broken_fields = summary.get('broken_fields', [])
        
        if broken_fields:
            for bf in broken_fields:
                severity = bf.get('severity', 5)
                
                if severity >= 8:
                    alert_type = st.error
                    icon = "🔴"
                elif severity >= 5:
                    alert_type = st.warning
                    icon = "🟡"
                else:
                    alert_type = st.info
                    icon = "🔵"
                
                with st.expander(f"{icon} {bf.get('column', 'Unknown')} (Severity: {severity}/10)"):
                    st.markdown(f"**Reason:** {bf.get('reason', 'N/A')}")
                    
                    if 'zero_pct' in bf:
                        st.markdown(f"- Zero values: {bf['zero_pct']:.1f}%")
                    if 'null_pct' in bf:
                        st.markdown(f"- Null values: {bf['null_pct']:.1f}%")
                    if 'variance' in bf:
                        st.markdown(f"- Variance: {bf['variance']:.6f}")
        else:
            st.success("✅ No broken fields detected!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Aadhaar Data Quality Intelligence Engine | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
