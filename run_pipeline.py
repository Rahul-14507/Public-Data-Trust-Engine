#!/usr/bin/env python
"""
Aadhaar Data Quality Intelligence Engine
=========================================
CLI entrypoint for running the complete data quality pipeline.

Usage:
    python run_pipeline.py --input-dir c:/Projects/UIDAI
    python run_pipeline.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from normalizer import DataNormalizer
from quality_checks import QualityChecker
from scoring import ReliabilityScorer
from reporter import AnomalyReporter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Aadhaar Data Quality Intelligence Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py --input-dir c:/Projects/UIDAI
    python run_pipeline.py --input-dir . --output-dir ./reports
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(Path(__file__).parent.parent),
        help='Directory containing Aadhaar dataset folders (default: parent of script directory)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).parent / 'outputs'),
        help='Directory to save output reports (default: ./outputs)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    return parser.parse_args()


def run_pipeline(input_dir: str, output_dir: str) -> dict:
    """
    Run the complete data quality pipeline.
    
    Args:
        input_dir: Path to directory containing dataset folders
        output_dir: Path to save output files
        
    Returns:
        Dictionary with pipeline results
    """
    logger.info("=" * 70)
    logger.info("AADHAAR DATA QUALITY INTELLIGENCE ENGINE")
    logger.info("=" * 70)
    logger.info(f"Input Directory: {input_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("")
    
    results = {}
    
    # =========================================================================
    # STEP 1: DATA INGESTION
    # =========================================================================
    logger.info("STEP 1: Loading datasets...")
    
    loader = DataLoader(input_dir)
    datasets = loader.load_all_datasets()
    
    if not datasets:
        logger.error("No datasets loaded! Check input directory.")
        return {'error': 'No datasets loaded'}
    
    results['row_counts'] = loader.get_row_counts()
    
    # =========================================================================
    # STEP 2: DATA NORMALIZATION
    # =========================================================================
    logger.info("\nSTEP 2: Normalizing data...")
    
    normalizer = DataNormalizer()
    normalized_datasets = {}
    
    for name, df in datasets.items():
        normalized_datasets[name] = normalizer.normalize_dataset(df, name)
    
    results['normalizer_stats'] = normalizer.get_stats()
    results['parsing_errors'] = normalizer.get_parsing_errors()
    
    # =========================================================================
    # STEP 3: QUALITY CHECKS
    # =========================================================================
    logger.info("\nSTEP 3: Running quality checks...")
    
    checker = QualityChecker()
    quality_results, anomalies_df = checker.run_all_checks(normalized_datasets)
    
    results['quality_metrics'] = checker.get_metrics()
    results['anomalies_count'] = len(anomalies_df)
    
    # =========================================================================
    # STEP 4: RELIABILITY SCORING
    # =========================================================================
    logger.info("\nSTEP 4: Calculating reliability score...")
    
    scorer = ReliabilityScorer()
    reliability_score = scorer.calculate_score(
        checker.get_metrics(),
        normalizer.get_stats()
    )
    
    results['reliability_score'] = reliability_score
    results['grade'] = scorer.get_grade()
    results['health_status'] = scorer.get_health_status()
    results['score_breakdown'] = scorer.get_breakdown()
    
    # =========================================================================
    # STEP 5: GENERATE REPORTS
    # =========================================================================
    logger.info("\nSTEP 5: Generating reports...")
    
    reporter = AnomalyReporter(output_dir)
    output_files = reporter.generate_all_reports(
        quality_results=quality_results,
        anomalies_df=anomalies_df,
        score_breakdown=scorer.get_breakdown(),
        reliability_score=reliability_score,
        health_status=scorer.get_health_status(),
        grade=scorer.get_grade(),
        row_counts=loader.get_row_counts(),
        normalizer_stats=normalizer.get_stats()
    )
    
    results['output_files'] = output_files
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Reliability Score: {reliability_score:.1f}/100 (Grade: {scorer.get_grade()})")
    logger.info(f"Health Status: {scorer.get_health_status()}")
    logger.info(f"Total Anomalies: {len(anomalies_df):,}")
    logger.info("")
    logger.info("Output Files:")
    for report_type, path in output_files.items():
        logger.info(f"  - {report_type}: {path}")
    logger.info("=" * 70)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = run_pipeline(args.input_dir, args.output_dir)
        
        if 'error' in results:
            sys.exit(1)
        
        # Exit with code based on reliability score
        score = results.get('reliability_score', 0)
        if score < 50:
            logger.warning("Exit code 2: Critical data quality issues")
            sys.exit(2)
        elif score < 70:
            logger.warning("Exit code 1: Data quality concerns")
            sys.exit(1)
        else:
            logger.info("Exit code 0: Data quality acceptable")
            sys.exit(0)
            
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
