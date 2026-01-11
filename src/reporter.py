"""
Anomaly Reporter Module
=======================
Generates machine-readable and human-readable reports.
Outputs: anomalies_summary.json, anomalies_table.csv, data_quality_report.md
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyReporter:
    """
    Generates data quality reports in multiple formats.
    
    Outputs:
    - anomalies_summary.json: Machine-readable summary
    - anomalies_table.csv: Row-level anomalies
    - data_quality_report.md: Human-readable report
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the AnomalyReporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AnomalyReporter initialized. Output directory: {self.output_dir}")
    
    def generate_all_reports(
        self,
        quality_results: List[Any],
        anomalies_df: pd.DataFrame,
        score_breakdown: Dict[str, Any],
        reliability_score: float,
        health_status: str,
        grade: str,
        row_counts: Dict[str, Any],
        normalizer_stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate all report files.
        
        Returns:
            Dictionary mapping report type to file path
        """
        logger.info("=" * 60)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 60)
        
        output_files = {}
        
        # Generate JSON summary
        json_path = self._generate_json_summary(
            quality_results, anomalies_df, score_breakdown,
            reliability_score, health_status, grade, row_counts
        )
        output_files['json'] = str(json_path)
        
        # Generate CSV table
        csv_path = self._generate_csv_table(anomalies_df)
        output_files['csv'] = str(csv_path)
        
        # Generate Markdown report
        md_path = self._generate_markdown_report(
            quality_results, anomalies_df, score_breakdown,
            reliability_score, health_status, grade, row_counts, normalizer_stats
        )
        output_files['markdown'] = str(md_path)
        
        logger.info(f"Reports generated in: {self.output_dir}")
        
        return output_files
    
    def _generate_json_summary(
        self,
        quality_results: List[Any],
        anomalies_df: pd.DataFrame,
        score_breakdown: Dict[str, Any],
        reliability_score: float,
        health_status: str,
        grade: str,
        row_counts: Dict[str, Any]
    ) -> Path:
        """Generate anomalies_summary.json."""
        
        # Process quality results
        broken_fields = []
        duplicate_stats = {}
        temporal_issues = {}
        cross_dataset = {}
        
        for result in quality_results:
            if result.check_name == 'broken_fields':
                broken_fields.extend(result.details.get('broken_fields', []))
            elif result.check_name == 'duplicates':
                ds = result.details.get('dataset', 'unknown')
                duplicate_stats[ds] = {
                    'exact_duplicates': int(result.details.get('exact_duplicates', 0)),
                    'exact_duplicate_pct': result.details.get('exact_duplicate_pct', 0),
                    'near_duplicates': int(result.details.get('near_duplicates', 0))
                }
            elif result.check_name == 'temporal_integrity':
                ds = result.details.get('dataset', 'unknown')
                temporal_issues[ds] = {
                    'date_range': result.details.get('date_range', ''),
                    'missing_dates': result.details.get('missing_dates', 0),
                    'sudden_changes': len(result.details.get('sudden_changes', []))
                }
            elif result.check_name == 'cross_dataset_consistency':
                cross_dataset = result.details.get('comparisons', [])
        
        # Count outliers by dataset
        outlier_counts = {}
        if not anomalies_df.empty and 'anomaly_type' in anomalies_df.columns:
            outlier_df = anomalies_df[anomalies_df['anomaly_type'] == 'outlier']
            if 'dataset' in outlier_df.columns:
                outlier_counts = outlier_df.groupby('dataset').size().to_dict()
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'health_score': round(reliability_score, 2),
            'grade': grade,
            'health_status': health_status,
            'score_breakdown': score_breakdown,
            'datasets': {
                name: {
                    'total_rows': counts.get('total', 0),
                    'files': counts.get('files', {})
                }
                for name, counts in row_counts.items()
            },
            'broken_fields': broken_fields,
            'duplicate_stats': duplicate_stats,
            'outlier_counts': outlier_counts,
            'temporal_issues': temporal_issues,
            'cross_dataset_inconsistencies': cross_dataset,
            'total_anomalies': len(anomalies_df) if not anomalies_df.empty else 0
        }
        
        output_path = self.output_dir / 'anomalies_summary.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"  Generated: {output_path}")
        return output_path
    
    def _generate_csv_table(self, anomalies_df: pd.DataFrame) -> Path:
        """Generate anomalies_table.csv with row-level anomaly flags."""
        
        output_path = self.output_dir / 'anomalies_table.csv'
        
        if anomalies_df.empty:
            # Create empty file with headers
            empty_df = pd.DataFrame(columns=[
                'dataset', 'row_index', 'anomaly_type', 'severity', 'details'
            ])
            empty_df.to_csv(output_path, index=False)
        else:
            # Ensure required columns exist
            required_cols = ['dataset', 'row_index', 'anomaly_type', 'severity', 'details']
            for col in required_cols:
                if col not in anomalies_df.columns:
                    anomalies_df[col] = ''
            
            # Select and order columns
            output_df = anomalies_df[required_cols].copy()
            output_df = output_df.sort_values(['severity', 'dataset'], ascending=[False, True])
            output_df.to_csv(output_path, index=False)
        
        logger.info(f"  Generated: {output_path}")
        return output_path
    
    def _generate_markdown_report(
        self,
        quality_results: List[Any],
        anomalies_df: pd.DataFrame,
        score_breakdown: Dict[str, Any],
        reliability_score: float,
        health_status: str,
        grade: str,
        row_counts: Dict[str, Any],
        normalizer_stats: Dict[str, Any]
    ) -> Path:
        """Generate data_quality_report.md - human-readable report."""
        
        output_path = self.output_dir / 'data_quality_report.md'
        
        lines = []
        
        # Header
        lines.append("# Aadhaar Data Quality Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        
        # Score display with emoji
        if reliability_score >= 80:
            emoji = "✅"
        elif reliability_score >= 60:
            emoji = "⚠️"
        else:
            emoji = "❌"
        
        lines.append(f"### Health Score: {emoji} {reliability_score:.1f}/100 (Grade: {grade})")
        lines.append("")
        lines.append(f"**Status:** {health_status}")
        lines.append("")
        
        # Dataset Overview
        lines.append("## Dataset Overview")
        lines.append("")
        lines.append("| Dataset | Total Rows | Files |")
        lines.append("|---------|-----------|-------|")
        
        total_rows = 0
        for name, counts in row_counts.items():
            rows = counts.get('total', 0)
            files = len(counts.get('files', {}))
            lines.append(f"| {name.capitalize()} | {rows:,} | {files} |")
            total_rows += rows
        
        lines.append(f"| **Total** | **{total_rows:,}** | - |")
        lines.append("")
        
        # Score Breakdown
        lines.append("## Score Breakdown")
        lines.append("")
        lines.append("| Penalty Category | Penalty Applied |")
        lines.append("|------------------|-----------------|")
        
        penalties = score_breakdown.get('penalties', {})
        for category, penalty in penalties.items():
            lines.append(f"| {category.replace('_', ' ').title()} | -{penalty:.2f} |")
        
        lines.append(f"| **Total Penalty** | **-{score_breakdown.get('total_penalty', 0):.2f}** |")
        lines.append("")
        
        # Quality Check Results
        lines.append("## Quality Check Results")
        lines.append("")
        
        for result in quality_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"### {result.check_name.replace('_', ' ').title()}: {status}")
            lines.append("")
            
            if result.check_name == 'broken_fields':
                broken = result.details.get('broken_fields', [])
                if broken:
                    lines.append("**Potentially Broken Fields:**")
                    lines.append("")
                    for bf in broken[:10]:
                        lines.append(f"- `{bf['column']}`: {bf['reason']}")
                    if len(broken) > 10:
                        lines.append(f"- _...and {len(broken) - 10} more_")
                else:
                    lines.append("No broken fields detected.")
                lines.append("")
                
            elif result.check_name == 'duplicates':
                lines.append(f"- **Dataset:** {result.details.get('dataset', 'N/A')}")
                lines.append(f"- **Exact Duplicates:** {result.details.get('exact_duplicates', 0):,} ({result.details.get('exact_duplicate_pct', 0):.2f}%)")
                lines.append(f"- **Near Duplicates:** {result.details.get('near_duplicates', 0):,}")
                lines.append("")
                
            elif result.check_name == 'outliers':
                lines.append(f"- **Dataset:** {result.details.get('dataset', 'N/A')}")
                lines.append(f"- **Total Outliers:** {result.details.get('total_outliers', 0):,}")
                lines.append(f"- **Outlier Percentage:** {result.details.get('outlier_pct', 0):.2f}%")
                lines.append("")
                
            elif result.check_name == 'temporal_integrity':
                lines.append(f"- **Dataset:** {result.details.get('dataset', 'N/A')}")
                lines.append(f"- **Date Range:** {result.details.get('date_range', 'N/A')}")
                lines.append(f"- **Missing Dates:** {result.details.get('missing_dates', 0)}")
                lines.append(f"- **Sudden Changes:** {len(result.details.get('sudden_changes', []))}")
                lines.append("")
                
            elif result.check_name == 'cross_dataset_consistency':
                comparisons = result.details.get('comparisons', [])
                for comp in comparisons:
                    lines.append(f"- **{comp['comparison'].replace('_', ' ').title()}:** {comp['abnormal_count']} abnormal ({comp['abnormal_pct']:.2f}%)")
                lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        
        recommendations = self._generate_recommendations(quality_results, reliability_score)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        lines.append("")
        
        # Risk Assessment
        lines.append("## Risk Assessment")
        lines.append("")
        
        if reliability_score >= 80:
            lines.append("🟢 **LOW RISK:** Data quality is acceptable for most use cases.")
        elif reliability_score >= 60:
            lines.append("🟡 **MEDIUM RISK:** Data quality issues may affect analysis accuracy. Review highlighted concerns before critical decisions.")
        else:
            lines.append("🔴 **HIGH RISK:** Significant data quality problems detected. Immediate investigation recommended before using this data.")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Aadhaar Data Quality Intelligence Engine*")
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"  Generated: {output_path}")
        return output_path
    
    def _generate_recommendations(
        self,
        quality_results: List[Any],
        reliability_score: float
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        
        recommendations = []
        
        for result in quality_results:
            if not result.passed:
                if result.check_name == 'broken_fields':
                    recommendations.append(
                        "**Investigate broken fields:** Review data pipeline for columns with high zero rates or near-zero variance. These may indicate extraction failures."
                    )
                elif result.check_name == 'duplicates':
                    if result.details.get('exact_duplicate_pct', 0) > 5:
                        recommendations.append(
                            "**Address duplicate records:** Implement deduplication in the ingestion pipeline to prevent duplicate entries."
                        )
                elif result.check_name == 'outliers':
                    if result.details.get('outlier_pct', 0) > 5:
                        recommendations.append(
                            "**Review outlier records:** Investigate high-value outliers to determine if they are data entry errors or legitimate exceptions."
                        )
                elif result.check_name == 'temporal_integrity':
                    if result.details.get('missing_dates', 0) > 0:
                        recommendations.append(
                            "**Fill temporal gaps:** Missing dates may indicate incomplete data extraction. Verify source system availability for gap periods."
                        )
                elif result.check_name == 'cross_dataset_consistency':
                    recommendations.append(
                        "**Align cross-dataset metrics:** Abnormal ratios between enrolment and update datasets may indicate mismatched reporting periods."
                    )
        
        if reliability_score < 70:
            recommendations.append(
                "**Conduct root cause analysis:** Low overall score warrants a comprehensive review of data collection and processing systems."
            )
        
        if not recommendations:
            recommendations.append(
                "**Maintain current practices:** Data quality is satisfactory. Continue monitoring for any degradation."
            )
        
        return recommendations
