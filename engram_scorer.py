#!/usr/bin/env python3
"""
Engram Layout Scorer for scoring keyboard layouts.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

Script for scoring keyboard layouts based on letter- and letter-pair frequencies,
and key- and key-pair comfort scores. Provides both 32-key (full layout) and 
24-key (home block only) scoring modes.

24-key scoring focuses on the home block keys: qwerasdfzxcvuiopjkl;m,./
This is what this scorer was originally designed for:
https://github.com/binarybottle/optimize_layouts
32-key scoring uses an extended set of keys based on a single comfort score estimate 
for all non-home-block keys (value greater than that of any hurdle, such as RV).

Usage:
    # Basic scoring (both 32-key and 24-key results)
    python engram_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

    # With cross-hand filtering (ignores different-hand pairs)
    python engram_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --csv --ignore-cross-hand

    # CSV output
    python engram_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --csv

    # Score only
    python engram_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --score-only

Required input files:
    - normalized_letter_frequencies_en.csv:               
      letter,frequency                           
      e,12.70   
    - normalized_letter_pair_frequencies_en.csv:
      letter_pair,frequency
      th,3.56
    - normalized_key_comfort_scores_24keys.csv:
      key,comfort_score
      D,7.2
    - normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv:
      key_pair,comfort_score
      DF,7.2

For reference:
    QWERTY: "qwertyuiopasdfghjkl;zxcvbnm,./"  
    Dvorak: "',.pyfgcrlaoeuidhtns;qjkxbmwvz"
    Colemak: "qwfpgjluy;arstdhneiozxcvbkm,./"
"""
import sys
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np

# Set to True to enable automatic distribution detection and normalization
do_detect_and_normalize_distribution = False  

# Import our framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config
from framework.layout_utils import filter_to_letters_only, is_same_hand_pair, get_layout_statistics
from framework.data_utils import load_csv_with_validation, validate_data_consistency
from framework.output_utils import print_results
from framework.cli_utils import create_standard_parser, handle_common_errors, get_layout_from_args

# Define the 24 home block keys (on, above, and below home rows)
HOME_BLOCK_KEYS = set('qwerasdfzxcvuiopjkl;m,./'.upper())

# Default combination strategy for scores
DEFAULT_COMBINATION_STRATEGY = "multiplicative"  # item_score * item_pair_score


def detect_and_normalize_distribution(scores: np.ndarray, name: str = '', verbose: bool = True) -> np.ndarray:
    """
    Automatically detect distribution type and apply appropriate normalization.
    Returns scores normalized to [0,1] range.
    """
    # Handle empty or constant arrays
    if len(scores) == 0 or np.all(scores == scores[0]):
        return np.zeros_like(scores)

    # Get basic statistics
    non_zeros = scores[scores != 0]
    if len(non_zeros) == 0:
        return np.zeros_like(scores)
        
    min_nonzero = np.min(non_zeros)
    max_val = np.max(scores)
    mean = np.mean(non_zeros)
    median = np.median(non_zeros)
    skew = np.mean(((non_zeros - mean) / np.std(non_zeros)) ** 3) if np.std(non_zeros) > 0 else 0
    
    # Calculate ratio between consecutive sorted values
    sorted_nonzero = np.sort(non_zeros)
    ratios = sorted_nonzero[1:] / sorted_nonzero[:-1] if len(sorted_nonzero) > 1 else np.array([1.0])
    
    # Detect distribution type and apply appropriate normalization
    if len(scores[scores == 0]) / len(scores) > 0.3:
        # Sparse distribution with many zeros
        if verbose:
            print(f"  {name}: Sparse distribution detected")
        norm_scores = np.sqrt(scores)
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))

    elif skew > 2 or np.median(ratios) > 1.5:
        # Heavy-tailed/exponential/zipfian distribution
        if verbose:
            print(f"  {name}: Heavy-tailed distribution detected")
        norm_scores = np.sqrt(np.abs(scores))
        return (norm_scores - np.min(norm_scores)) / (np.max(norm_scores) - np.min(norm_scores))
        
    elif abs(mean - median) / mean < 0.1:
        # Roughly symmetric distribution
        if verbose:
            print(f"  {name}: Symmetric distribution detected")
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
    else:
        # Default to robust scaling
        if verbose:
            print(f"  {name}: Using robust scaling")
        q1, q99 = np.percentile(scores, [1, 99])
        scaled = (scores - q1) / (q99 - q1)
        return np.clip(scaled, 0, 1)


def apply_combination_strategy(item_score: float, item_pair_score: float, strategy: str = "multiplicative") -> float:
    """Apply the specified combination strategy for item and item-pair scores."""
    if strategy == "multiplicative":
        return item_score * item_pair_score
    elif strategy == "additive":
        return item_score + item_pair_score
    elif strategy == "weighted_additive":
        return 0.25 * item_score + 0.75 * item_pair_score
    else:
        raise ValueError(f"Unknown combination strategy: {strategy}")


def filter_to_home_block(layout_mapping: Dict[str, str]) -> Dict[str, str]:
    """Filter layout mapping to only include home block keys."""
    filtered = {}
    for char, pos in layout_mapping.items():
        if pos.upper() in HOME_BLOCK_KEYS:
            filtered[char] = pos
    return filtered


class EngramScorer(BaseLayoutScorer):
    """
    Engram Layout Scorer using the unified framework.
    
    Scores keyboard layouts based on letter frequencies and key comfort scores
    using multiplicative combination of item frequencies and position comfort.
    Provides both 32-key (full layout) and 24-key (home block only) scoring.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Engram scorer.
        
        Args:
            layout_mapping: Dict mapping characters to positions
            config: Optional configuration dictionary
        """
        super().__init__(layout_mapping, config)
        
        # Data containers
        self.item_scores: Dict[str, float] = {}
        self.item_pair_scores: Dict[Tuple[str, str], float] = {}
        self.position_scores: Dict[str, float] = {}
        self.position_pair_scores: Dict[Tuple[str, str], float] = {}
        
        # Get scoring options from config
        scoring_options = self.config.get('scoring_options', {})
        self.combination_strategy = scoring_options.get('combination_strategy', DEFAULT_COMBINATION_STRATEGY)
        self.normalization_method = scoring_options.get('normalization_method', 'auto')
        self.ignore_cross_hand = scoring_options.get('ignore_cross_hand', False)
        
        # Get combination weights for weighted additive
        self.item_weight = scoring_options.get('item_weight', 0.6)
        self.item_pair_weight = scoring_options.get('item_pair_weight', 0.4)
        
        # Create filtered layout for 24-key scoring
        self.layout_mapping_24key = filter_to_home_block(self.layout_mapping)
    
    def load_data_files(self) -> None:
        """Load and normalize all score files."""
        data_files = self.config.get('data_files', {})
        quiet_mode = self.config.get('quiet_mode', False)
        
        if not quiet_mode:
            print("Loading and normalizing score files...")
        
        # Load and normalize item scores
        item_file = data_files.get('item_scores')
        if not item_file or not Path(item_file).exists():
            raise FileNotFoundError(f"Item scores file required: {item_file}")
        
        if not quiet_mode:
            print(f"  Loading item scores from: {item_file}")
        
        item_df = load_csv_with_validation(item_file, ['letter', 'frequency'])
        scores = item_df['frequency'].astype(float).values
        if do_detect_and_normalize_distribution:
            norm_scores = detect_and_normalize_distribution(scores, 'Item scores', not quiet_mode)
        else:
            norm_scores = scores

        self.item_scores = {}
        for i, (_, row) in enumerate(item_df.iterrows()):
            letter = str(row['letter']).lower()
            self.item_scores[letter] = float(norm_scores[i])
        
        # Load and normalize item pair scores
        item_pair_file = data_files.get('item_pair_scores')
        if not item_pair_file or not Path(item_pair_file).exists():
            raise FileNotFoundError(f"Item pair scores file required: {item_pair_file}")
        
        if not quiet_mode:
            print(f"  Loading item pair scores from: {item_pair_file}")
        
        item_pair_df = load_csv_with_validation(item_pair_file, ['letter_pair', 'frequency'])
        scores = item_pair_df['frequency'].astype(float).values
        if do_detect_and_normalize_distribution:
            norm_scores = detect_and_normalize_distribution(scores, 'Item pair scores', not quiet_mode)
        else:
            norm_scores = scores

        self.item_pair_scores = {}
        for i, (_, row) in enumerate(item_pair_df.iterrows()):
            pair_str = str(row['letter_pair'])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                self.item_pair_scores[key] = float(norm_scores[i])
        
        # Load and normalize position scores
        position_file = data_files.get('position_scores')
        if not position_file or not Path(position_file).exists():
            raise FileNotFoundError(f"Position scores file required: {position_file}")
        
        if not quiet_mode:
            print(f"  Loading position scores from: {position_file}")
        
        position_df = load_csv_with_validation(position_file, ['key', 'comfort_score'])
        scores = position_df['comfort_score'].astype(float).values
        if do_detect_and_normalize_distribution:
            norm_scores = detect_and_normalize_distribution(scores, 'Position scores', not quiet_mode)
        else:
            norm_scores = scores

        self.position_scores = {}
        for i, (_, row) in enumerate(position_df.iterrows()):
            key = str(row['key']).lower()
            self.position_scores[key] = float(norm_scores[i])
        
        # Load and normalize position-pair scores
        position_pair_file = data_files.get('position_pair_scores')
        if not position_pair_file or not Path(position_pair_file).exists():
            raise FileNotFoundError(f"Position-pair scores file required: {position_pair_file}")
        
        if not quiet_mode:
            print(f"  Loading position-pair scores from: {position_pair_file}")
        
        position_pair_df = load_csv_with_validation(position_pair_file, ['key_pair', 'comfort_score'])
        scores = position_pair_df['comfort_score'].astype(float).values
        if do_detect_and_normalize_distribution:
            norm_scores = detect_and_normalize_distribution(scores, 'Position-pair scores', not quiet_mode)
        else:
            norm_scores = scores

        self.position_pair_scores = {}
        for i, (_, row) in enumerate(position_pair_df.iterrows()):
            pair_str = str(row['key_pair'])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                self.position_pair_scores[key] = float(norm_scores[i])
        
        if not quiet_mode:
            print(f"  Loaded {len(self.item_scores)} item scores")
            print(f"  Loaded {len(self.item_pair_scores)} item pair scores")
            print(f"  Loaded {len(self.position_scores)} position scores")
            print(f"  Loaded {len(self.position_pair_scores)} position-pair scores")
            
            # Show 24-key coverage
            print(f"  24-key home block coverage: {len(self.layout_mapping_24key)}/{len(self.layout_mapping)} keys")
        
        # Validate position coverage
        self._validate_position_coverage(quiet_mode)
    
    def _validate_position_coverage(self, quiet_mode: bool) -> None:
        """Validate that positions have corresponding entries in the position-pair file."""
        if quiet_mode:
            return
            
        used_positions = set(pos.lower() for pos in self.layout_mapping.values())
        
        # Get positions that appear in the position-pair file
        available_positions = set()
        for (pos1, pos2) in self.position_pair_scores.keys():
            available_positions.add(pos1)
            available_positions.add(pos2)
        
        # Check for missing individual positions
        missing_positions = used_positions - available_positions
        
        if missing_positions:
            missing_str = ''.join(sorted(missing_positions)).upper()
            available_str = ''.join(sorted(available_positions)).upper()
            print(f"  Warning: Missing positions in pair file: {missing_str}")
            print(f"  Available positions: {available_str}")
        else:
            print(f"  ✓ All positions found in position-pair file")
    
    def calculate_layout_score(self, layout_mapping: Dict[str, str], score_name: str = "32key") -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Calculate complete layout score for given layout mapping.
        
        Args:
            layout_mapping: Character to position mapping to score
            score_name: Name for this scoring mode (for logging)
        
        Returns:
            Tuple of (total_score, item_component, item_pair_component, filtering_info)
        """
        items = list(layout_mapping.keys())
        positions = list(layout_mapping.values())
        n_items = len(items)
        
        # Calculate item component
        item_raw_score = 0.0
        for item, pos in zip(items, positions):
            item_score = self.item_scores.get(item.lower(), 0.0)    # DEFAULT to 0.0 if not found
            pos_score = self.position_scores.get(pos.lower(), 0.0)  # DEFAULT to 0.0 if not found
            item_raw_score += item_score * pos_score
        
        item_component = item_raw_score / n_items if n_items > 0 else 0.0
        
        # Calculate item-pair component with optional filtering
        pair_raw_score = 0.0
        pair_count = 0
        cross_hand_pairs_filtered = 0
        
        for i in range(n_items):
            for j in range(n_items):
                if i != j:  # Skip self-pairs
                    item1, item2 = items[i].lower(), items[j].lower()
                    pos1, pos2 = positions[i].lower(), positions[j].lower()
                    
                    # Filter cross-hand pairs if requested
                    if self.ignore_cross_hand and not is_same_hand_pair(pos1, pos2):
                        cross_hand_pairs_filtered += 1
                        continue
                    
                    # Get scores with defaults
                    item_pair_key = (item1, item2)
                    item_pair_score = self.item_pair_scores.get(item_pair_key, 1.0)    # DEFAULT to 1.0 if not found

                    pos_pair_key = (pos1, pos2)
                    pos_pair_score = self.position_pair_scores.get(pos_pair_key, 1.0)  # DEFAULT to 1.0 if not found

                    pair_raw_score += item_pair_score * pos_pair_score
                    pair_count += 1
        
        pair_component = pair_raw_score / max(1, pair_count)
        
        # Apply combination strategy
        if self.combination_strategy == "weighted_additive":
            total_score = self.item_weight * item_component + self.item_pair_weight * pair_component
        else:
            total_score = apply_combination_strategy(item_component, pair_component, self.combination_strategy)
        
        filtering_info = {
            'cross_hand_pairs_filtered': cross_hand_pairs_filtered,
            'pairs_used': pair_count,
            'total_possible_pairs': n_items * (n_items - 1),
            'scoring_mode': score_name
        }
        
        return total_score, item_component, pair_component, filtering_info
    
    def calculate_scores(self) -> ScoreResult:
        """
        Calculate layout scores using both 32-key and 24-key Engram methodology.
        
        Returns:
            ScoreResult containing both scoring modes, components, and metadata
        """
        # Calculate 32-key scores (full layout)
        total_score_32, item_component_32, pair_component_32, filtering_info_32 = self.calculate_layout_score(
            self.layout_mapping, "32key"
        )
        
        # Calculate 24-key scores (home block only)
        total_score_24, item_component_24, pair_component_24, filtering_info_24 = self.calculate_layout_score(
            self.layout_mapping_24key, "24key"
        )
        
        # Get layout statistics
        layout_stats = get_layout_statistics(self.layout_mapping)
        layout_stats_24 = get_layout_statistics(self.layout_mapping_24key)
        
        # Use 32-key score as primary
        primary_score = total_score_32
        
        # Create result with both scoring modes
        result = ScoreResult(
            primary_score=primary_score,
            components={
                # 32-key components
                'total_score_32key': total_score_32,
                'item_component_32key': item_component_32,
                'item_pair_component_32key': pair_component_32,
                # 24-key components
                'total_score_24key': total_score_24,
                'item_component_24key': item_component_24,
                'item_pair_component_24key': pair_component_24,
                # Legacy components for compatibility
                'item_component': item_component_32,
                'item_pair_component': pair_component_32,
            },
            metadata={
                'combination_strategy': self.combination_strategy,
                'normalization_method': self.normalization_method,
                'ignore_cross_hand': self.ignore_cross_hand,
                'layout_size_32key': len(self.layout_mapping),
                'layout_size_24key': len(self.layout_mapping_24key),
                'alphabet_coverage_32key': layout_stats.get('alphabet_coverage', 0.0),
                'alphabet_coverage_24key': layout_stats_24.get('alphabet_coverage', 0.0),
                'home_block_coverage': len(self.layout_mapping_24key) / len(self.layout_mapping) if self.layout_mapping else 0.0,
            },
            validation_info={
                'total_chars_32key': layout_stats['total_chars'],
                'total_chars_24key': layout_stats_24['total_chars'],
                'letters_32key': layout_stats['letters'],
                'letters_24key': layout_stats_24['letters'],
                'alphabet_coverage_percentage_32key': layout_stats.get('alphabet_coverage', 0.0) * 100,
                'alphabet_coverage_percentage_24key': layout_stats_24.get('alphabet_coverage', 0.0) * 100,
                'cross_hand_filtering': self.ignore_cross_hand,
            },
            detailed_breakdown={
                'filtering_info_32key': filtering_info_32,
                'filtering_info_24key': filtering_info_24,
                'data_coverage': {
                    'item_scores_loaded': len(self.item_scores),
                    'item_pair_scores_loaded': len(self.item_pair_scores),
                    'position_scores_loaded': len(self.position_scores),
                    'position_pair_scores_loaded': len(self.position_pair_scores),
                },
                'scoring_options': {
                    'combination_strategy': self.combination_strategy,
                    'item_weight': self.item_weight,
                    'item_pair_weight': self.item_pair_weight,
                },
                'home_block_keys': sorted(list(HOME_BLOCK_KEYS)),
                'layout_24key_mapping': dict(self.layout_mapping_24key),
            }
        )
        
        return result


@handle_common_errors
def main() -> int:
    """Main entry point using the standardized framework."""
    
    # Create standardized CLI parser
    cli_parser = create_standard_parser('engram_scorer')
    args = cli_parser.parse_args()
    
    try:
        # Load configuration
        config = load_scorer_config('engram_scorer', args.config)
        
        # Override with command-line arguments
        config['quiet_mode'] = args.quiet
        
        # Handle cross-hand filtering
        if hasattr(args, 'ignore_cross_hand') and args.ignore_cross_hand:
            if 'scoring_options' not in config:
                config['scoring_options'] = {}
            config['scoring_options']['ignore_cross_hand'] = True
        
        # Get layout mapping from arguments
        letters, positions, layout_mapping = get_layout_from_args(args)
        
        # Filter to letters only (same as original behavior)
        layout_mapping = filter_to_letters_only(layout_mapping)
        
        if not layout_mapping:
            print("Error: No letters found in layout")
            return 1
        
        # Create and run scorer
        scorer = EngramScorer(layout_mapping, config)
        result = scorer.score_layout()
        
        # Print results using framework output utilities
        output_format = args.output_format
        output_config = config.get('output_formats', {}).get(output_format, {})
        
        print_results(result, output_format, output_config)
        
        return 0
        
    except Exception as e:
        if not args.quiet:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())