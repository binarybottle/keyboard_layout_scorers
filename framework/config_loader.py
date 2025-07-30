#!/usr/bin/env python3
"""
Configuration loader for keyboard layout scorers.

Provides unified configuration management using YAML files.
Handles merging of common settings with scorer-specific settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import os


class ConfigLoader:
    """Handles loading and processing of YAML configuration files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load YAML configuration file with validation.
        
        Returns:
            Full configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if self._config_cache is not None:
            return self._config_cache
            
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
        
        if config is None:
            config = {}
            
        self._config_cache = config
        return config

    def get_scorer_config(self, scorer_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific scorer with common settings merged.
        
        Args:
            scorer_name: Name of the scorer (e.g., 'distance_scorer')
            
        Returns:
            Merged configuration dictionary for the scorer
            
        Raises:
            ValueError: If scorer not found in configuration
        """
        full_config = self.load_config()
        
        if scorer_name not in full_config:
            available_scorers = [k for k in full_config.keys() 
                               if k not in ['common', 'output_formats', 'cli']]
            raise ValueError(
                f"Scorer '{scorer_name}' not found in configuration. "
                f"Available scorers: {available_scorers}"
            )
        
        # Get common settings
        common_config = full_config.get('common', {})
        scorer_config = full_config[scorer_name].copy()
        
        # Merge data directories and resolve file paths
        self._resolve_data_file_paths(scorer_config, common_config, scorer_name)
        
        # Merge common settings (scorer-specific settings take precedence)
        merged_config = {**common_config, **scorer_config}
        
        return merged_config

    def _resolve_data_file_paths(self, scorer_config: Dict[str, Any], 
                                common_config: Dict[str, Any], 
                                scorer_name: str) -> None:
        """
        Resolve relative data file paths using directory configuration.
        
        Args:
            scorer_config: Scorer-specific configuration (modified in place)
            common_config: Common configuration settings
            scorer_name: Name of the scorer
        """
        data_directories = common_config.get('data_directories', {})
        
        if 'data_files' not in scorer_config:
            return
            
        # Determine the appropriate data directory
        base_dir = data_directories.get('base', 'input/')
        scorer_key = scorer_name.replace('_scorer', '')
        scorer_dir = data_directories.get(scorer_key, base_dir)
        
        # Resolve each data file path
        for key, filename in scorer_config['data_files'].items():
            if filename is None:
                continue
                
            filepath = Path(filename)
            
            # If path is relative and doesn't already start with the base directory
            if not filepath.is_absolute() and not str(filepath).startswith(base_dir):
                resolved_path = Path(scorer_dir) / filename
                scorer_config['data_files'][key] = str(resolved_path)

    def get_output_format_config(self, format_name: str) -> Dict[str, Any]:
        """
        Get output format configuration.
        
        Args:
            format_name: Name of output format (csv, detailed, score_only)
            
        Returns:
            Output format configuration
        """
        full_config = self.load_config()
        output_formats = full_config.get('output_formats', {})
        
        return output_formats.get(format_name, {})

    def get_cli_config(self) -> Dict[str, Any]:
        """
        Get CLI argument configuration.
        
        Returns:
            CLI configuration with argument mappings
        """
        full_config = self.load_config()
        return full_config.get('cli', {})

    def get_available_scorers(self) -> List[str]:
        """
        Get list of available scorer names.
        
        Returns:
            List of scorer names found in configuration
        """
        full_config = self.load_config()
        
        # Exclude special sections that aren't scorers
        excluded_sections = {'common', 'output_formats', 'cli'}
        
        return [k for k in full_config.keys() if k not in excluded_sections]

    def validate_scorer_config(self, scorer_name: str) -> List[str]:
        """
        Validate a scorer's configuration and return any issues found.
        
        Args:
            scorer_name: Name of the scorer to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            config = self.get_scorer_config(scorer_name)
        except (ValueError, FileNotFoundError, yaml.YAMLError) as e:
            return [f"Configuration error: {e}"]
        
        issues = []
        
        # Check required sections
        required_sections = ['description', 'method', 'output']
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Check output configuration
        if 'output' in config:
            output_config = config['output']
            if 'primary_score_name' not in output_config:
                issues.append("Missing primary_score_name in output configuration")
        
        # Check data file existence (if specified)
        if 'data_files' in config:
            for file_key, filepath in config['data_files'].items():
                if filepath is not None and not Path(filepath).exists():
                    issues.append(f"Data file not found: {file_key} -> {filepath}")
        
        return issues


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None

def get_config_loader(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Get global configuration loader instance (singleton pattern).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None or _config_loader.config_path != Path(config_path):
        _config_loader = ConfigLoader(config_path)
    
    return _config_loader

def load_scorer_config(scorer_name: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Convenience function to load configuration for a specific scorer.
    
    Args:
        scorer_name: Name of the scorer
        config_path: Path to configuration file
        
    Returns:
        Scorer configuration dictionary
    """
    loader = get_config_loader(config_path)
    return loader.get_scorer_config(scorer_name)