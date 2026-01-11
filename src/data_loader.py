"""
Data Loader Module
==================
Handles ingestion of Aadhaar datasets from CSV files.
Maintains raw copies and logs row counts per dataset.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and manages Aadhaar datasets from CSV files.
    
    Attributes:
        input_dir: Base directory containing dataset folders
        datasets: Dictionary mapping dataset names to DataFrames
        raw_copies: Dictionary of untouched raw DataFrames
        row_counts: Dictionary of row counts per file and dataset
    """
    
    DATASET_FOLDERS = {
        'enrolment': 'api_data_aadhar_enrolment',
        'demographic': 'api_data_aadhar_demographic',
        'biometric': 'api_data_aadhar_biometric'
    }
    
    def __init__(self, input_dir: str):
        """
        Initialize the DataLoader.
        
        Args:
            input_dir: Path to the directory containing dataset folders
        """
        self.input_dir = Path(input_dir)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.raw_copies: Dict[str, pd.DataFrame] = {}
        self.row_counts: Dict[str, Dict[str, int]] = {}
        
        logger.info(f"DataLoader initialized with input directory: {self.input_dir}")
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all three Aadhaar datasets.
        
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        for name, folder in self.DATASET_FOLDERS.items():
            self.load_dataset(name, folder)
        
        self._log_summary()
        return self.datasets
    
    def load_dataset(self, name: str, folder: str) -> Optional[pd.DataFrame]:
        """
        Load a single dataset from its folder.
        
        Args:
            name: Logical name for the dataset (e.g., 'enrolment')
            folder: Folder name containing CSV files
            
        Returns:
            Combined DataFrame or None if loading fails
        """
        folder_path = self.input_dir / folder
        
        if not folder_path.exists():
            logger.error(f"Dataset folder not found: {folder_path}")
            return None
        
        # Find all CSV files in the folder
        csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {folder_path}")
            return None
        
        logger.info(f"Loading {name} dataset from {len(csv_files)} files...")
        
        # Load and concatenate all CSV files
        dfs = []
        file_counts = {}
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['_source_file'] = os.path.basename(csv_file)
                df['_dataset'] = name
                
                file_counts[os.path.basename(csv_file)] = len(df)
                dfs.append(df)
                
                logger.info(f"  Loaded {os.path.basename(csv_file)}: {len(df):,} rows")
                
            except Exception as e:
                logger.error(f"  Error loading {csv_file}: {e}")
        
        if not dfs:
            return None
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Store dataset and raw copy
        self.datasets[name] = combined_df
        self.raw_copies[name] = combined_df.copy()
        
        # Store row counts
        self.row_counts[name] = {
            'files': file_counts,
            'total': len(combined_df)
        }
        
        logger.info(f"  Total {name} rows: {len(combined_df):,}")
        
        return combined_df
    
    def get_raw_copy(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get an untouched raw copy of a dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Raw DataFrame copy or None if not loaded
        """
        return self.raw_copies.get(name)
    
    def get_row_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get row counts for all loaded datasets.
        
        Returns:
            Dictionary of row counts per dataset and file
        """
        return self.row_counts
    
    def _log_summary(self) -> None:
        """Log summary of all loaded datasets."""
        logger.info("=" * 60)
        logger.info("DATA LOADING SUMMARY")
        logger.info("=" * 60)
        
        total_rows = 0
        for name, counts in self.row_counts.items():
            logger.info(f"{name.upper()}: {counts['total']:,} rows from {len(counts['files'])} files")
            total_rows += counts['total']
        
        logger.info(f"TOTAL ROWS LOADED: {total_rows:,}")
        logger.info("=" * 60)
