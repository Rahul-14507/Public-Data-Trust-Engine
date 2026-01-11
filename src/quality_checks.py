"""
Quality Checks Module
=====================
Implements comprehensive data quality checks:
A) Broken field detection
B) Duplicate detection
C) Outlier detection (IQR & Z-score)
D) Temporal integrity checks
E) Cross-dataset consistency checks
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Container for quality check results."""
    check_name: str
    passed: bool
    severity: int  # 1-10 scale
    details: Dict[str, Any] = field(default_factory=dict)
    affected_rows: List[int] = field(default_factory=list)


class QualityChecker:
    """
    Performs comprehensive data quality checks on Aadhaar datasets.
    
    Attributes:
        results: List of QualityCheckResult objects
        anomalies: DataFrame of row-level anomalies
    """
    
    def __init__(self):
        """Initialize the QualityChecker."""
        self.results: List[QualityCheckResult] = []
        self.anomalies: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
    
    def run_all_checks(
        self,
        datasets: Dict[str, pd.DataFrame],
        normalized: bool = True
    ) -> Tuple[List[QualityCheckResult], pd.DataFrame]:
        """
        Run all quality checks on the datasets.
        
        Args:
            datasets: Dictionary of dataset name to DataFrame
            normalized: Whether datasets have been normalized
            
        Returns:
            Tuple of (results list, anomalies DataFrame)
        """
        logger.info("=" * 60)
        logger.info("RUNNING DATA QUALITY CHECKS")
        logger.info("=" * 60)
        
        for name, df in datasets.items():
            logger.info(f"\nChecking {name} dataset ({len(df):,} rows)...")
            
            # A) Broken field detection
            self._check_broken_fields(df, name)
            
            # B) Duplicate detection
            self._check_duplicates(df, name)
            
            # C) Outlier detection
            self._check_outliers(df, name)
            
            # D) Temporal integrity
            if 'date' in df.columns:
                self._check_temporal_integrity(df, name)
        
        # E) Cross-dataset consistency (if all datasets present)
        if all(k in datasets for k in ['enrolment', 'demographic', 'biometric']):
            self._check_cross_dataset_consistency(
                datasets['enrolment'],
                datasets['demographic'],
                datasets['biometric']
            )
        
        # Convert anomalies to DataFrame
        anomalies_df = pd.DataFrame(self.anomalies) if self.anomalies else pd.DataFrame()
        
        self._log_summary()
        
        return self.results, anomalies_df
    
    # =========================================================================
    # A) BROKEN FIELD DETECTION
    # =========================================================================
    
    def _check_broken_fields(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Detect broken or suspicious numeric fields.
        
        Flags fields where:
        - Zero percentage > 90%
        - Variance is near zero (< 0.001)
        - High null percentage (> 50%)
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
        """
        logger.info("  [A] Checking for broken fields...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude internal columns
        numeric_cols = [c for c in numeric_cols if not c.startswith('_')]
        
        broken_fields = []
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            total = len(df)
            
            if len(col_data) == 0:
                broken_fields.append({
                    'column': col,
                    'reason': 'All null values',
                    'severity': 10
                })
                continue
            
            # Calculate metrics
            zero_pct = (col_data == 0).sum() / total * 100
            null_pct = df[col].isna().sum() / total * 100
            variance = col_data.var() if len(col_data) > 1 else 0
            
            # Calculate entropy (normalized)
            try:
                value_counts = col_data.value_counts(normalize=True)
                entropy = stats.entropy(value_counts) if len(value_counts) > 1 else 0
            except:
                entropy = 0
            
            # Check for broken field conditions
            is_broken = False
            reasons = []
            severity = 0
            
            if zero_pct > 90:
                is_broken = True
                reasons.append(f"Zero values: {zero_pct:.1f}%")
                severity = max(severity, 8)
            
            if variance < 0.001 and len(col_data) > 100:
                is_broken = True
                reasons.append(f"Near-zero variance: {variance:.6f}")
                severity = max(severity, 7)
            
            if null_pct > 50:
                is_broken = True
                reasons.append(f"High null rate: {null_pct:.1f}%")
                severity = max(severity, 6)
            
            if is_broken:
                broken_fields.append({
                    'column': col,
                    'reason': '; '.join(reasons),
                    'severity': severity,
                    'zero_pct': zero_pct,
                    'null_pct': null_pct,
                    'variance': variance,
                    'entropy': entropy
                })
        
        # Store result
        self.results.append(QualityCheckResult(
            check_name='broken_fields',
            passed=len(broken_fields) == 0,
            severity=max([f['severity'] for f in broken_fields]) if broken_fields else 0,
            details={
                'dataset': dataset_name,
                'broken_fields': broken_fields,
                'total_checked': len(numeric_cols)
            }
        ))
        
        self.metrics[f'{dataset_name}_broken_fields'] = len(broken_fields)
        
        if broken_fields:
            logger.warning(f"    Found {len(broken_fields)} potentially broken fields")
            for bf in broken_fields:
                logger.warning(f"      - {bf['column']}: {bf['reason']}")
        else:
            logger.info(f"    All {len(numeric_cols)} numeric fields healthy")
    
    # =========================================================================
    # B) DUPLICATE DETECTION
    # =========================================================================
    
    def _check_duplicates(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Detect exact and near-duplicate rows.
        
        Exact duplicates: Identical rows
        Near-duplicates: Same date + pincode + similar values
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
        """
        logger.info("  [B] Checking for duplicates...")
        
        total_rows = len(df)
        
        # Exclude internal columns for duplicate checking
        check_cols = [c for c in df.columns if not c.startswith('_')]
        
        # Exact duplicates
        exact_dups = df.duplicated(subset=check_cols, keep=False)
        exact_dup_count = exact_dups.sum()
        exact_dup_pct = (exact_dup_count / total_rows) * 100
        
        # Near-duplicates: same date + pincode + district
        key_cols = ['date', 'pincode', 'district']
        available_key_cols = [c for c in key_cols if c in df.columns]
        
        if available_key_cols:
            near_dups = df.duplicated(subset=available_key_cols, keep=False)
            near_dup_count = near_dups.sum()
            near_dup_pct = (near_dup_count / total_rows) * 100
        else:
            near_dup_count = 0
            near_dup_pct = 0
        
        # Record anomalies for duplicate rows
        if exact_dup_count > 0:
            dup_indices = df[exact_dups].index.tolist()[:100]  # Sample
            for idx in dup_indices:
                self.anomalies.append({
                    'dataset': dataset_name,
                    'row_index': idx,
                    'anomaly_type': 'exact_duplicate',
                    'severity': 5,
                    'details': 'Exact duplicate row detected'
                })
        
        # Determine severity
        if exact_dup_pct > 20:
            severity = 9
        elif exact_dup_pct > 10:
            severity = 7
        elif exact_dup_pct > 5:
            severity = 5
        elif exact_dup_pct > 0:
            severity = 3
        else:
            severity = 0
        
        self.results.append(QualityCheckResult(
            check_name='duplicates',
            passed=exact_dup_count == 0,
            severity=severity,
            details={
                'dataset': dataset_name,
                'exact_duplicates': exact_dup_count,
                'exact_duplicate_pct': exact_dup_pct,
                'near_duplicates': near_dup_count,
                'near_duplicate_pct': near_dup_pct,
                'total_rows': total_rows
            }
        ))
        
        self.metrics[f'{dataset_name}_duplicate_pct'] = exact_dup_pct
        self.metrics[f'{dataset_name}_near_duplicate_pct'] = near_dup_pct
        
        logger.info(f"    Exact duplicates: {exact_dup_count:,} ({exact_dup_pct:.2f}%)")
        logger.info(f"    Near duplicates: {near_dup_count:,} ({near_dup_pct:.2f}%)")
    
    # =========================================================================
    # C) OUTLIER DETECTION
    # =========================================================================
    
    def _check_outliers(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Detect outliers using IQR and Z-score methods.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
        """
        logger.info("  [C] Checking for outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.startswith('_')]
        
        all_outliers = []
        
        for col in numeric_cols:
            # IQR method
            iqr_outliers = self._detect_outliers_iqr(df, col)
            
            # Z-score method
            zscore_outliers = self._detect_outliers_zscore(df, col)
            
            # Combine (union of both methods)
            combined = set(iqr_outliers) | set(zscore_outliers)
            
            for idx in list(combined)[:50]:  # Limit to 50 per column
                all_outliers.append({
                    'dataset': dataset_name,
                    'row_index': idx,
                    'column': col,
                    'value': df.loc[idx, col] if idx in df.index else None,
                    'method': 'iqr+zscore'
                })
                
                self.anomalies.append({
                    'dataset': dataset_name,
                    'row_index': idx,
                    'anomaly_type': 'outlier',
                    'severity': 4,
                    'details': f'Outlier in {col}'
                })
        
        total_outliers = len(all_outliers)
        outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
        
        self.results.append(QualityCheckResult(
            check_name='outliers',
            passed=outlier_pct < 5,
            severity=min(10, int(outlier_pct)),
            details={
                'dataset': dataset_name,
                'total_outliers': total_outliers,
                'outlier_pct': outlier_pct,
                'top_outliers': all_outliers[:50]
            }
        ))
        
        self.metrics[f'{dataset_name}_outlier_count'] = total_outliers
        self.metrics[f'{dataset_name}_outlier_pct'] = outlier_pct
        
        logger.info(f"    Total outliers detected: {total_outliers:,} ({outlier_pct:.2f}%)")
    
    def _detect_outliers_iqr(
        self, 
        df: pd.DataFrame, 
        column: str,
        multiplier: float = 1.5
    ) -> List[int]:
        """
        Detect outliers using the IQR method.
        
        Args:
            df: Input DataFrame
            column: Column to check
            multiplier: IQR multiplier (default 1.5)
            
        Returns:
            List of row indices containing outliers
        """
        data = df[column].dropna()
        if len(data) < 4:
            return []
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        return df[outlier_mask].index.tolist()
    
    def _detect_outliers_zscore(
        self, 
        df: pd.DataFrame, 
        column: str,
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect outliers using Z-score method.
        
        Args:
            df: Input DataFrame
            column: Column to check
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            List of row indices containing outliers
        """
        data = df[column].dropna()
        if len(data) < 4:
            return []
        
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return []
        
        z_scores = np.abs((df[column] - mean) / std)
        outlier_mask = z_scores > threshold
        return df[outlier_mask].index.tolist()
    
    # =========================================================================
    # D) TEMPORAL INTEGRITY CHECKS
    # =========================================================================
    
    def _check_temporal_integrity(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Check temporal integrity of the data.
        
        Detects:
        - Missing date ranges
        - Sudden jumps or drops (>50% day-over-day)
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
        """
        logger.info("  [D] Checking temporal integrity...")
        
        # Parse dates
        df_temp = df.copy()
        df_temp['date_parsed'] = pd.to_datetime(df_temp['date'], errors='coerce')
        
        # Date parsing failures
        date_failures = df_temp['date_parsed'].isna().sum()
        
        # Get valid dates
        valid_dates = df_temp['date_parsed'].dropna()
        
        if len(valid_dates) == 0:
            logger.error("    No valid dates found!")
            self.metrics[f'{dataset_name}_temporal_gaps'] = 0
            return
        
        # Find date range
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        
        # Expected dates (daily)
        expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        actual_dates = set(valid_dates.dt.date)
        expected_dates_set = set(expected_dates.date)
        
        # Missing dates
        missing_dates = expected_dates_set - actual_dates
        gap_count = len(missing_dates)
        
        # Day-over-day volume changes
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if not c.startswith('_')]
        
        sudden_changes = []
        
        if numeric_cols:
            daily_sums = df_temp.groupby('date_parsed')[numeric_cols[0]].sum()
            daily_sums = daily_sums.sort_index()
            
            if len(daily_sums) > 1:
                pct_change = daily_sums.pct_change().abs()
                jumps = pct_change[pct_change > 0.5]  # >50% change
                
                for date, change in jumps.items():
                    if pd.notna(change) and change != float('inf'):
                        sudden_changes.append({
                            'date': str(date.date()) if hasattr(date, 'date') else str(date),
                            'pct_change': change * 100
                        })
                        
                        self.anomalies.append({
                            'dataset': dataset_name,
                            'row_index': -1,
                            'anomaly_type': 'temporal_jump',
                            'severity': 6,
                            'details': f'Sudden {change*100:.1f}% change on {date}'
                        })
        
        self.results.append(QualityCheckResult(
            check_name='temporal_integrity',
            passed=gap_count == 0 and len(sudden_changes) == 0,
            severity=min(10, gap_count // 5 + len(sudden_changes)),
            details={
                'dataset': dataset_name,
                'date_range': f"{min_date.date()} to {max_date.date()}",
                'missing_dates': gap_count,
                'date_parse_failures': date_failures,
                'sudden_changes': sudden_changes[:10],
                'sample_missing': list(missing_dates)[:10]
            }
        ))
        
        self.metrics[f'{dataset_name}_temporal_gaps'] = gap_count
        self.metrics[f'{dataset_name}_sudden_changes'] = len(sudden_changes)
        
        logger.info(f"    Date range: {min_date.date()} to {max_date.date()}")
        logger.info(f"    Missing dates: {gap_count}")
        logger.info(f"    Sudden volume changes: {len(sudden_changes)}")
    
    # =========================================================================
    # E) CROSS-DATASET CONSISTENCY
    # =========================================================================
    
    def _check_cross_dataset_consistency(
        self,
        enrol_df: pd.DataFrame,
        demo_df: pd.DataFrame,
        bio_df: pd.DataFrame
    ) -> None:
        """
        Check consistency across datasets.
        
        Compares:
        - Enrolment age_5_17 vs demographic demo_age_5_17
        - Enrolment age_5_17 vs biometric bio_age_5_17
        
        Args:
            enrol_df: Enrolment DataFrame
            demo_df: Demographic DataFrame
            bio_df: Biometric DataFrame
        """
        logger.info("\n  [E] Checking cross-dataset consistency...")
        
        inconsistencies = []
        
        # Aggregate by date and pincode for comparison
        key_cols = ['date', 'pincode']
        
        # Check if required columns exist
        enrol_cols = ['age_5_17'] if 'age_5_17' in enrol_df.columns else []
        demo_cols = ['demo_age_5_17'] if 'demo_age_5_17' in demo_df.columns else []
        bio_cols = ['bio_age_5_17'] if 'bio_age_5_17' in bio_df.columns else []
        
        if enrol_cols and demo_cols:
            # Compare enrolment vs demographic updates
            enrol_agg = enrol_df.groupby(key_cols)['age_5_17'].sum().reset_index()
            demo_agg = demo_df.groupby(key_cols)['demo_age_5_17'].sum().reset_index()
            
            merged = enrol_agg.merge(demo_agg, on=key_cols, how='inner')
            
            if len(merged) > 0:
                # Calculate ratio (demo updates should generally be <= enrolments)
                merged['ratio'] = merged['demo_age_5_17'] / merged['age_5_17'].replace(0, np.nan)
                
                # Flag abnormal ratios (outside 0.1 to 10)
                abnormal = merged[(merged['ratio'] < 0.1) | (merged['ratio'] > 10)]
                
                inconsistencies.append({
                    'comparison': 'enrolment_vs_demographic',
                    'total_matched': len(merged),
                    'abnormal_count': len(abnormal),
                    'abnormal_pct': (len(abnormal) / len(merged) * 100) if len(merged) > 0 else 0
                })
                
                logger.info(f"    Enrolment vs Demographic: {len(abnormal)} abnormal ratios out of {len(merged)}")
        
        if enrol_cols and bio_cols:
            # Compare enrolment vs biometric updates
            enrol_agg = enrol_df.groupby(key_cols)['age_5_17'].sum().reset_index()
            bio_agg = bio_df.groupby(key_cols)['bio_age_5_17'].sum().reset_index()
            
            merged = enrol_agg.merge(bio_agg, on=key_cols, how='inner')
            
            if len(merged) > 0:
                merged['ratio'] = merged['bio_age_5_17'] / merged['age_5_17'].replace(0, np.nan)
                abnormal = merged[(merged['ratio'] < 0.1) | (merged['ratio'] > 10)]
                
                inconsistencies.append({
                    'comparison': 'enrolment_vs_biometric',
                    'total_matched': len(merged),
                    'abnormal_count': len(abnormal),
                    'abnormal_pct': (len(abnormal) / len(merged) * 100) if len(merged) > 0 else 0
                })
                
                logger.info(f"    Enrolment vs Biometric: {len(abnormal)} abnormal ratios out of {len(merged)}")
        
        total_abnormal = sum(i['abnormal_count'] for i in inconsistencies)
        
        self.results.append(QualityCheckResult(
            check_name='cross_dataset_consistency',
            passed=total_abnormal == 0,
            severity=min(10, total_abnormal // 100),
            details={
                'comparisons': inconsistencies
            }
        ))
        
        self.metrics['cross_dataset_abnormal'] = total_abnormal
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all quality check metrics."""
        return self.metrics
    
    def _log_summary(self) -> None:
        """Log summary of all quality checks."""
        logger.info("\n" + "=" * 60)
        logger.info("QUALITY CHECK SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        logger.info(f"Total checks: {len(self.results)}")
        logger.info(f"Passed: {passed}, Failed: {failed}")
        logger.info(f"Total anomalies recorded: {len(self.anomalies)}")
        logger.info("=" * 60)
