#!/usr/bin/env python3
"""
Factory for creating scorer instances.
"""

from typing import Dict, Any, List
from framework.base_scorer import BaseLayoutScorer


class ScorerFactory:
    """Factory for creating scorer instances."""
    
    SCORERS = {
        'distance': 'distance_scorer.DistanceScorer',
        'dvorak9': 'dvorak9_scorer.Dvorak9Scorer',
        'engram': 'engram_scorer.EngramScorer',
    }
    
    @classmethod
    def create_scorer(cls, scorer_name: str, layout_mapping: Dict[str, str], 
                     config: Dict[str, Any]) -> BaseLayoutScorer:
        """
        Create a scorer instance.
        
        Args:
            scorer_name: Name of scorer ('distance', 'dvorak9', 'engram')
            layout_mapping: Character to position mapping
            config: Configuration dictionary
            
        Returns:
            Configured scorer instance
            
        Raises:
            ValueError: If scorer_name is not recognized
        """
        if scorer_name not in cls.SCORERS:
            available = list(cls.SCORERS.keys())
            raise ValueError(f"Unknown scorer '{scorer_name}'. Available: {available}")
        
        # Dynamic import to avoid circular dependencies
        module_path, class_name = cls.SCORERS[scorer_name].rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        scorer_class = getattr(module, class_name)
        
        return scorer_class(layout_mapping, config)
    
    @classmethod
    def get_available_scorers(cls) -> List[str]:
        """Get list of available scorer names."""
        return list(cls.SCORERS.keys())