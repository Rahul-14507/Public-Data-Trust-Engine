"""
Data Normalizer Module
======================
Standardizes dates, validates pincodes, and normalizes numeric columns.
Tracks all parsing failures for quality reporting.
"""

import re
import logging
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalizes Aadhaar datasets for consistent analysis.
    
    Attributes:
        parsing_errors: Dictionary tracking all parsing failures
        stats: Dictionary of normalization statistics
    """
    
    PINCODE_PATTERN = re.compile(r'^\d{6}$')
    
    def __init__(self):
        """Initialize the DataNormalizer."""
        self.parsing_errors: Dict[str, List[Dict[str, Any]]] = {
            'date_errors': [],
            'pincode_errors': [],
            'numeric_errors': []
        }
        self.stats: Dict[str, Any] = {}
    
    def normalize_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Apply all normalizations to a dataset.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset for error tracking
            
        Returns:
            Normalized DataFrame
        """
        logger.info(f"Normalizing {dataset_name} dataset...")
        
        df = df.copy()
        
        # Normalize date column
        if 'date' in df.columns:
            df = self._normalize_dates(df, dataset_name)
        
        # Validate pincodes
        if 'pincode' in df.columns:
            df = self._validate_pincodes(df, dataset_name)
        
        # Normalize numeric columns
        numeric_cols = self._get_numeric_columns(df)
        df = self._normalize_numeric_columns(df, numeric_cols, dataset_name)
        
        self._log_stats(dataset_name)
        
        return df
    
    def _normalize_dates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Standardize date column to ISO format (YYYY-MM-DD).
        Handles mixed formats safely using dayfirst=True.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with normalized dates
        """
        original_count = len(df)
        
        # Store original dates for error tracking
        original_dates = df['date'].copy()
        
        # Parse dates with flexible format handling
        # Using dayfirst=True since data appears to use DD-MM-YYYY or D/M/YYYY
        df['date_normalized'] = pd.to_datetime(
            df['date'], 
            dayfirst=True, 
            errors='coerce'
        )
        
        # Track parsing failures
        failed_mask = df['date_normalized'].isna() & df['date'].notna()
        failed_count = failed_mask.sum()
        
        if failed_count > 0:
            failed_dates = original_dates[failed_mask].unique()[:50]  # Sample
            self.parsing_errors['date_errors'].extend([
                {
                    'dataset': dataset_name,
                    'value': str(d),
                    'error': 'Failed to parse date'
                }
                for d in failed_dates
            ])
            logger.warning(f"  Date parsing failed for {failed_count:,} rows ({failed_count/original_count*100:.2f}%)")
        
        # Convert to ISO format string
        df['date'] = df['date_normalized'].dt.strftime('%Y-%m-%d')
        df.drop('date_normalized', axis=1, inplace=True)
        
        # Store stats
        self.stats[f'{dataset_name}_date_errors'] = failed_count
        self.stats[f'{dataset_name}_date_error_pct'] = (failed_count / original_count) * 100
        
        success_count = original_count - failed_count
        logger.info(f"  Dates normalized: {success_count:,} success, {failed_count:,} failed")
        
        return df
    
    def _validate_pincodes(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Validate pincodes are 6-digit numeric values.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with pincode validation flag
        """
        original_count = len(df)
        
        # Convert to string for validation
        df['pincode_str'] = df['pincode'].astype(str)
        
        # Validate using regex
        df['pincode_valid'] = df['pincode_str'].str.match(self.PINCODE_PATTERN, na=False)
        
        # Track invalid pincodes
        invalid_mask = ~df['pincode_valid']
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            invalid_pincodes = df.loc[invalid_mask, 'pincode'].unique()[:50]
            self.parsing_errors['pincode_errors'].extend([
                {
                    'dataset': dataset_name,
                    'value': str(p),
                    'error': 'Invalid pincode format (expected 6 digits)'
                }
                for p in invalid_pincodes
            ])
            logger.warning(f"  Invalid pincodes: {invalid_count:,} rows ({invalid_count/original_count*100:.2f}%)")
        
        # Clean up temporary column
        df.drop('pincode_str', axis=1, inplace=True)
        
        # Store stats
        self.stats[f'{dataset_name}_pincode_errors'] = invalid_count
        self.stats[f'{dataset_name}_pincode_error_pct'] = (invalid_count / original_count) * 100
        
        valid_count = original_count - invalid_count
        logger.info(f"  Pincodes validated: {valid_count:,} valid, {invalid_count:,} invalid")
        
        return df
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify columns that should be numeric (age/count columns).
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names that should be numeric
        """
        # Columns that contain age or count data
        numeric_patterns = ['age_', 'demo_', 'bio_']
        
        numeric_cols = []
        for col in df.columns:
            if any(pattern in col.lower() for pattern in numeric_patterns):
                numeric_cols.append(col)
        
        return numeric_cols
    
    def _normalize_numeric_columns(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Convert numeric columns safely, coercing invalid values to NaN.
        
        Args:
            df: Input DataFrame
            columns: List of columns to convert
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with normalized numeric columns
        """
        total_errors = 0
        
        for col in columns:
            if col not in df.columns:
                continue
            
            original_non_null = df[col].notna().sum()
            
            # Convert to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Count conversion failures
            new_non_null = df[col].notna().sum()
            errors = original_non_null - new_non_null
            
            if errors > 0:
                total_errors += errors
                self.parsing_errors['numeric_errors'].append({
                    'dataset': dataset_name,
                    'column': col,
                    'error_count': errors
                })
        
        if total_errors > 0:
            logger.warning(f"  Numeric conversion errors: {total_errors:,} values coerced to NaN")
        
        self.stats[f'{dataset_name}_numeric_errors'] = total_errors
        logger.info(f"  Numeric columns normalized: {len(columns)} columns processed")
        
        return df
    
    def get_parsing_errors(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all parsing errors encountered during normalization.
        
        Returns:
            Dictionary of parsing errors by type
        """
        return self.parsing_errors
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get normalization statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats
    
    def _log_stats(self, dataset_name: str) -> None:
        """Log normalization statistics for a dataset."""
        logger.info(f"  Normalization complete for {dataset_name}")
