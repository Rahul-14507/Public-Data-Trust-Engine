"""
Reliability Scoring Module
==========================
Calculates a dataset reliability score (0-100) based on quality metrics.
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Breakdown of the reliability score calculation."""
    base_score: int = 100
    duplicate_penalty: float = 0
    broken_fields_penalty: float = 0
    outlier_penalty: float = 0
    date_error_penalty: float = 0
    missing_data_penalty: float = 0
    temporal_gap_penalty: float = 0
    cross_dataset_penalty: float = 0
    final_score: float = 100
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'base_score': self.base_score,
            'penalties': {
                'duplicate': self.duplicate_penalty,
                'broken_fields': self.broken_fields_penalty,
                'outlier': self.outlier_penalty,
                'date_error': self.date_error_penalty,
                'missing_data': self.missing_data_penalty,
                'temporal_gap': self.temporal_gap_penalty,
                'cross_dataset': self.cross_dataset_penalty
            },
            'total_penalty': self.base_score - self.final_score,
            'final_score': self.final_score
        }


class ReliabilityScorer:
    """
    Calculates reliability score for Aadhaar datasets.
    
    Scoring Formula (start with 100 points):
    -----------------------------------------
    
    1. Duplicate Penalty: -(duplicate_pct * 2)
       - Max penalty: 20 points
       - Rationale: Duplicates indicate ingestion issues
    
    2. Broken Fields Penalty: -(broken_count * 5)
       - Max penalty: 25 points
       - Rationale: Broken fields indicate pipeline failures
    
    3. Outlier Penalty: -(outlier_pct * 1.5)
       - Max penalty: 15 points
       - Rationale: High outliers may indicate data corruption
    
    4. Date Error Penalty: -(date_error_pct * 2)
       - Max penalty: 15 points
       - Rationale: Date errors affect temporal analysis
    
    5. Missing Data Penalty: -(null_pct * 1)
       - Max penalty: 15 points
       - Rationale: Missing data reduces completeness
    
    6. Temporal Gap Penalty: -(gap_days * 0.5)
       - Max penalty: 10 points
       - Rationale: Missing dates affect continuity
    
    Final Score = max(0, 100 - total_penalty)
    """
    
    # Penalty weights and caps
    DUPLICATE_WEIGHT = 2.0
    DUPLICATE_CAP = 20.0
    
    BROKEN_FIELDS_WEIGHT = 5.0
    BROKEN_FIELDS_CAP = 25.0
    
    OUTLIER_WEIGHT = 1.5
    OUTLIER_CAP = 15.0
    
    DATE_ERROR_WEIGHT = 2.0
    DATE_ERROR_CAP = 15.0
    
    MISSING_DATA_WEIGHT = 1.0
    MISSING_DATA_CAP = 15.0
    
    TEMPORAL_GAP_WEIGHT = 0.5
    TEMPORAL_GAP_CAP = 10.0
    
    CROSS_DATASET_WEIGHT = 0.1
    CROSS_DATASET_CAP = 10.0
    
    def __init__(self):
        """Initialize the ReliabilityScorer."""
        self.scores: Dict[str, ScoreBreakdown] = {}
        self.overall_score: float = 100.0
        self.overall_breakdown: ScoreBreakdown = ScoreBreakdown()
    
    def calculate_score(
        self,
        quality_metrics: Dict[str, Any],
        normalizer_stats: Dict[str, Any]
    ) -> float:
        """
        Calculate the overall reliability score.
        
        Args:
            quality_metrics: Metrics from QualityChecker
            normalizer_stats: Statistics from DataNormalizer
            
        Returns:
            Reliability score (0-100)
        """
        logger.info("=" * 60)
        logger.info("CALCULATING RELIABILITY SCORE")
        logger.info("=" * 60)
        
        breakdown = ScoreBreakdown()
        
        # 1. Duplicate Penalty
        # -------------------
        # Aggregate duplicate percentages across all datasets
        dup_pcts = [v for k, v in quality_metrics.items() if 'duplicate_pct' in k]
        avg_dup_pct = sum(dup_pcts) / len(dup_pcts) if dup_pcts else 0
        breakdown.duplicate_penalty = min(
            avg_dup_pct * self.DUPLICATE_WEIGHT,
            self.DUPLICATE_CAP
        )
        logger.info(f"  Duplicate penalty: {breakdown.duplicate_penalty:.2f} (avg {avg_dup_pct:.2f}%)")
        
        # 2. Broken Fields Penalty
        # ------------------------
        broken_counts = [v for k, v in quality_metrics.items() if 'broken_fields' in k]
        total_broken = sum(broken_counts)
        breakdown.broken_fields_penalty = min(
            total_broken * self.BROKEN_FIELDS_WEIGHT,
            self.BROKEN_FIELDS_CAP
        )
        logger.info(f"  Broken fields penalty: {breakdown.broken_fields_penalty:.2f} ({total_broken} fields)")
        
        # 3. Outlier Penalty
        # ------------------
        outlier_pcts = [v for k, v in quality_metrics.items() if 'outlier_pct' in k]
        avg_outlier_pct = sum(outlier_pcts) / len(outlier_pcts) if outlier_pcts else 0
        breakdown.outlier_penalty = min(
            avg_outlier_pct * self.OUTLIER_WEIGHT,
            self.OUTLIER_CAP
        )
        logger.info(f"  Outlier penalty: {breakdown.outlier_penalty:.2f} (avg {avg_outlier_pct:.2f}%)")
        
        # 4. Date Error Penalty
        # ---------------------
        date_error_pcts = [v for k, v in normalizer_stats.items() if 'date_error_pct' in k]
        avg_date_error = sum(date_error_pcts) / len(date_error_pcts) if date_error_pcts else 0
        breakdown.date_error_penalty = min(
            avg_date_error * self.DATE_ERROR_WEIGHT,
            self.DATE_ERROR_CAP
        )
        logger.info(f"  Date error penalty: {breakdown.date_error_penalty:.2f} (avg {avg_date_error:.2f}%)")
        
        # 5. Missing Data Penalty
        # -----------------------
        # Calculate from normalizer stats (pincode and numeric errors as proxy)
        pincode_error_pcts = [v for k, v in normalizer_stats.items() if 'pincode_error_pct' in k]
        avg_missing = sum(pincode_error_pcts) / len(pincode_error_pcts) if pincode_error_pcts else 0
        breakdown.missing_data_penalty = min(
            avg_missing * self.MISSING_DATA_WEIGHT,
            self.MISSING_DATA_CAP
        )
        logger.info(f"  Missing data penalty: {breakdown.missing_data_penalty:.2f} (avg {avg_missing:.2f}%)")
        
        # 6. Temporal Gap Penalty
        # -----------------------
        gap_counts = [v for k, v in quality_metrics.items() if 'temporal_gaps' in k]
        total_gaps = sum(gap_counts)
        breakdown.temporal_gap_penalty = min(
            total_gaps * self.TEMPORAL_GAP_WEIGHT,
            self.TEMPORAL_GAP_CAP
        )
        logger.info(f"  Temporal gap penalty: {breakdown.temporal_gap_penalty:.2f} ({total_gaps} gaps)")
        
        # 7. Cross-Dataset Penalty
        # ------------------------
        cross_abnormal = quality_metrics.get('cross_dataset_abnormal', 0)
        breakdown.cross_dataset_penalty = min(
            cross_abnormal * self.CROSS_DATASET_WEIGHT,
            self.CROSS_DATASET_CAP
        )
        logger.info(f"  Cross-dataset penalty: {breakdown.cross_dataset_penalty:.2f} ({cross_abnormal} abnormal)")
        
        # Calculate Final Score
        # ---------------------
        total_penalty = (
            breakdown.duplicate_penalty +
            breakdown.broken_fields_penalty +
            breakdown.outlier_penalty +
            breakdown.date_error_penalty +
            breakdown.missing_data_penalty +
            breakdown.temporal_gap_penalty +
            breakdown.cross_dataset_penalty
        )
        
        breakdown.final_score = max(0, 100 - total_penalty)
        
        self.overall_breakdown = breakdown
        self.overall_score = breakdown.final_score
        
        logger.info("-" * 40)
        logger.info(f"  Total Penalty: {total_penalty:.2f}")
        logger.info(f"  RELIABILITY SCORE: {breakdown.final_score:.1f}/100")
        logger.info("=" * 60)
        
        return breakdown.final_score
    
    def get_breakdown(self) -> Dict[str, Any]:
        """Get the score breakdown as a dictionary."""
        return self.overall_breakdown.to_dict()
    
    def get_grade(self) -> str:
        """
        Get a letter grade based on the reliability score.
        
        Returns:
            Letter grade (A, B, C, D, F)
        """
        score = self.overall_score
        
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def get_health_status(self) -> str:
        """
        Get a health status description.
        
        Returns:
            Health status string
        """
        score = self.overall_score
        
        if score >= 90:
            return "Excellent - Data is highly reliable"
        elif score >= 80:
            return "Good - Minor quality issues detected"
        elif score >= 70:
            return "Fair - Some quality concerns require attention"
        elif score >= 60:
            return "Poor - Significant quality issues present"
        else:
            return "Critical - Data reliability is severely compromised"
