#!/usr/bin/env python3
"""
Base classes for keyboard layout scorers.

Provides common interface and result structures for all scoring methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import time


@dataclass
class ScoreResult:
    """
    Standardized result container for keyboard layout scoring.
    
    Provides a consistent interface for all scoring methods while allowing
    scorer-specific additional data.
    """
    
    # Core scores
    primary_score: float
    """Main score for this layout (higher = better unless noted otherwise)"""
    
    components: Dict[str, float] = field(default_factory=dict)
    """Individual score components (e.g., item_score, item_pair_score)"""
    
    # Metadata
    scorer_name: str = ""
    """Name of the scoring method used"""
    
    layout_mapping: Dict[str, str] = field(default_factory=dict)
    """Character to position mapping used for scoring"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional scorer-specific metadata"""
    
    # Analysis details
    detailed_breakdown: Dict[str, Any] = field(default_factory=dict)
    """Detailed analysis breakdown (for detailed output mode)"""
    
    validation_info: Dict[str, Any] = field(default_factory=dict)
    """Validation and coverage information"""
    
    # Execution info
    execution_time: float = 0.0
    """Time taken to calculate scores (seconds)"""
    
    config_used: Dict[str, Any] = field(default_factory=dict)
    """Configuration settings used for this scoring"""
    
    def get_score(self, component_name: Optional[str] = None) -> float:
        """
        Get a specific score component or the primary score.
        
        Args:
            component_name: Name of component score to retrieve, or None for primary
            
        Returns:
            Requested score value
            
        Raises:
            KeyError: If component_name not found in components
        """
        if component_name is None:
            return self.primary_score
        
        if component_name not in self.components:
            available = list(self.components.keys())
            raise KeyError(f"Component '{component_name}' not found. Available: {available}")
        
        return self.components[component_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.
        
        Returns:
            Dictionary representation suitable for JSON/CSV export
        """
        result = {
            'primary_score': self.primary_score,
            'scorer_name': self.scorer_name,
            'execution_time': self.execution_time,
        }
        
        # Add component scores with prefixed names
        for component, score in self.components.items():
            result[f'component_{component}'] = score
        
        # Add flattened metadata
        for key, value in self.metadata.items():
            # Skip complex objects that can't be easily serialized
            if isinstance(value, (str, int, float, bool)):
                result[f'meta_{key}'] = value
        
        # Add validation summary
        if self.validation_info:
            for key, value in self.validation_info.items():
                if isinstance(value, (str, int, float, bool)):
                    result[f'validation_{key}'] = value
        
        return result
    
    def summary(self) -> str:
        """
        Get a brief summary string of the results.
        
        Returns:
            Human-readable summary
        """
        summary_lines = [
            f"Scorer: {self.scorer_name}",
            f"Primary score: {self.primary_score:.6f}",
        ]
        
        if self.components:
            summary_lines.append("Components:")
            for name, score in self.components.items():
                summary_lines.append(f"  {name}: {score:.6f}")
        
        if self.execution_time > 0:
            summary_lines.append(f"Execution time: {self.execution_time:.3f}s")
        
        return "\n".join(summary_lines)


    def extract_all_metrics(self) -> Dict[str, float]:
        """
        Extract all numeric metrics from this result for detailed analysis.
        
        Returns:
            Dictionary of all available numeric metrics with prefixed names
        """
        metrics = {}
        
        # Always include the primary score
        metrics[f"{self.scorer_name}_primary"] = self.primary_score
        
        # Add component scores
        for component, score in self.components.items():
            metrics[f"{self.scorer_name}_{component}"] = score
        
        if 'distance' in self.scorer_name:
            # Distance scorer metrics - include per-finger distances from components
            for key, value in self.metadata.items():
                if isinstance(value, (int, float)) and not key.startswith('_'):
                    metrics[f"distance_{key}"] = float(value)
        
        elif 'dvorak9' in self.scorer_name:
            # Dvorak9 scorer - look for the 9 individual principle scores
            principle_names = [
                'same_hand_finger', 'same_finger', 'hand_alternation', 
                'finger_strength', 'row_jumping', 'home_row_usage',
                'outward_rolls', 'inward_rolls', 'lateral_movement'
            ]
            
            for principle in principle_names:
                if principle in self.metadata:
                    metrics[f"dvorak9_{principle}"] = float(self.metadata[principle])
            
            # Also check for any other numeric metadata
            for key, value in self.metadata.items():
                if isinstance(value, (int, float)) and key not in principle_names and not key.startswith('_'):
                    metrics[f"dvorak9_{key}"] = float(value)
        
        elif 'engram' in self.scorer_name:
            # Engram scorer metrics
            engram_metrics = ['item_score', 'item_pair_score', 'total_score', 'comfort_score', 'frequency_score']
            
            for metric in engram_metrics:
                if metric in self.metadata:
                    metrics[f"engram_{metric}"] = float(self.metadata[metric])
            
            # Also check for any other numeric metadata
            for key, value in self.metadata.items():
                if isinstance(value, (int, float)) and key not in engram_metrics and not key.startswith('_'):
                    metrics[f"engram_{key}"] = float(value)
        
        return metrics

class BaseLayoutScorer(ABC):
    """
    Abstract base class for keyboard layout scoring methods.
    
    Provides common interface and utilities that all scorers should implement.
    Handles configuration loading, layout validation, and result formatting.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base scorer.
        
        Args:
            layout_mapping: Dict mapping characters to positions (e.g., {'a': 'F'})
            config: Optional configuration dictionary (loaded from YAML if None)
        """
        self.layout_mapping = layout_mapping.copy()
        self.config = config or {}
        self.scorer_name = self.__class__.__name__.lower().replace('scorer', '_scorer')
        
        # Validate layout mapping
        self._validate_layout_mapping()
        
        # Load required data files
        self._data_loaded = False
        
    def _validate_layout_mapping(self) -> None:
        """
        Validate the layout mapping for basic correctness.
        
        Raises:
            ValueError: If layout mapping is invalid
        """
        if not self.layout_mapping:
            raise ValueError("Layout mapping cannot be empty")
        
        # Check for duplicate positions
        positions = list(self.layout_mapping.values())
        if len(positions) != len(set(positions)):
            duplicates = [pos for pos in set(positions) if positions.count(pos) > 1]
            raise ValueError(f"Duplicate positions in layout mapping: {duplicates}")
        
        # Check for empty keys or values
        for char, pos in self.layout_mapping.items():
            if not char or not pos:
                raise ValueError(f"Empty character or position in mapping: '{char}' -> '{pos}'")
    
    @abstractmethod
    def load_data_files(self) -> None:
        """
        Load required data files for scoring.
        
        This method should load any CSV files or other data needed for scoring.
        Should be implemented by each specific scorer.
        
        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data files have invalid format
        """
        pass
    
    @abstractmethod
    def calculate_scores(self) -> ScoreResult:
        """
        Calculate layout scores using the scorer's methodology.
        
        Returns:
            ScoreResult containing primary score, components, and metadata
        """
        pass
    
    def score_layout(self, load_data: bool = True) -> ScoreResult:
        """
        Main entry point for scoring a layout.
        
        Args:
            load_data: Whether to load data files if not already loaded
            
        Returns:
            ScoreResult with timing information
        """
        start_time = time.time()
        
        # Load data files if needed
        if load_data and not self._data_loaded:
            self.load_data_files()
            self._data_loaded = True
        
        # Calculate scores
        result = self.calculate_scores()
        
        # Add metadata
        result.execution_time = time.time() - start_time
        result.scorer_name = self.scorer_name
        result.layout_mapping = self.layout_mapping.copy()
        result.config_used = self.config.copy()
        
        return result
    
    def get_layout_string(self) -> str:
        """
        Get a string representation of the current layout.
        
        Returns:
            Layout mapping as a readable string
        """
        chars = ''.join(sorted(self.layout_mapping.keys()))
        positions = ''.join(self.layout_mapping[c] for c in sorted(self.layout_mapping.keys()))
        return f"{chars} → {positions}"
    
    def validate_data_files(self) -> List[str]:
        """
        Validate that required data files exist and are readable.
        
        Returns:
            List of validation error messages (empty if all files valid)
        """
        issues = []
        
        data_files = self.config.get('data_files', {})
        
        for file_key, filepath in data_files.items():
            if filepath is None:
                continue
                
            path_obj = Path(filepath)
            
            if not path_obj.exists():
                issues.append(f"Data file not found: {file_key} -> {filepath}")
            elif not path_obj.is_file():
                issues.append(f"Data file path is not a file: {file_key} -> {filepath}")
            elif not path_obj.suffix.lower() in ['.csv', '.txt', '.tsv']:
                issues.append(f"Unexpected file format for {file_key}: {filepath}")
        
        return issues
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary with key configuration details
        """
        return {
            'scorer_name': self.scorer_name,
            'layout_size': len(self.layout_mapping),
            'data_files_configured': len(self.config.get('data_files', {})),
            'scoring_options': self.config.get('scoring_options', {}),
            'output_config': self.config.get('output', {}),
        }