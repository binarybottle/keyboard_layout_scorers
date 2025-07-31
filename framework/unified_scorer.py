#!/usr/bin/env python3
"""
Unified manager for running and comparing keyboard layout scorers.
"""

from typing import Dict, Any, List
from framework.base_scorer import ScoreResult
from framework.config_loader import load_scorer_config, get_config_loader
from framework.scorer_factory import ScorerFactory


class UnifiedLayoutScorer:
    """
    Unified manager for running and comparing keyboard layout scorers.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the unified scorer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_loader = get_config_loader(config_path)
        self.factory = ScorerFactory()
    
    def score_layout(self, layout_mapping: Dict[str, str], 
                    scorers: List[str],
                    **kwargs) -> Dict[str, ScoreResult]:
        """
        Score a layout using specified scorers.
        
        Args:
            layout_mapping: Character to position mapping
            scorers: List of scorer names to run
            **kwargs: Additional arguments (text, weights, etc.)
            
        Returns:
            Dict mapping scorer names to their results
        """
        results = {}
        
        for scorer_name in scorers:
            try:
                # Load scorer-specific configuration
                config = load_scorer_config(f'{scorer_name}_scorer')
                
                # Add scorer-specific arguments
                if scorer_name == 'distance' and 'text' in kwargs:
                    config['text'] = kwargs['text']
                elif scorer_name == 'dvorak9' and 'weights' in kwargs:
                    config['weights_file'] = kwargs['weights']
                elif scorer_name == 'engram' and 'ignore_cross_hand' in kwargs:
                    if 'scoring_options' not in config:
                        config['scoring_options'] = {}
                    config['scoring_options']['ignore_cross_hand'] = kwargs['ignore_cross_hand']
                
                # Add quiet mode
                config['quiet_mode'] = kwargs.get('quiet', False)
                
                # Create and run scorer
                scorer = self.factory.create_scorer(scorer_name, layout_mapping, config)
                result = scorer.score_layout()
                results[scorer_name] = result
                
            except Exception as e:
                if not kwargs.get('quiet', False):
                    print(f"Warning: {scorer_name} scorer failed: {e}")
                # Create empty result for failed scorer
                results[scorer_name] = ScoreResult(
                    primary_score=0.0,
                    metadata={'error': str(e), 'scorer_failed': True}
                )
        
        return results
    
    def compare_layouts(self, layouts: Dict[str, Dict[str, str]], 
                       scorers: List[str],
                       **kwargs) -> Dict[str, Dict[str, ScoreResult]]:
        """
        Compare multiple layouts across specified scorers.
        
        Args:
            layouts: Dict mapping layout names to their character mappings
            scorers: List of scorer names to run
            **kwargs: Additional arguments
            
        Returns:
            Nested dict: {layout_name: {scorer_name: ScoreResult}}
        """
        results = {}
        
        for layout_name, layout_mapping in layouts.items():
            if not kwargs.get('quiet', False):
                print(f"Scoring layout: {layout_name}")
            
            results[layout_name] = self.score_layout(layout_mapping, scorers, **kwargs)
        
        return results