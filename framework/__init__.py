# framework/__init__.py
"""
Unified Keyboard Layout Scoring Framework

Common utilities and base classes for keyboard layout evaluation.
"""

__version__ = "2.0.0"

# Import main classes for easy access
from .base_scorer import BaseLayoutScorer, ScoreResult
from .config_loader import ConfigLoader, load_scorer_config

__all__ = [
    'BaseLayoutScorer', 
    'ScoreResult', 
    'ConfigLoader', 
    'load_scorer_config'
]