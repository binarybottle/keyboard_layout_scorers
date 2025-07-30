#!/usr/bin/env python3
"""
Distance Layout Scorer for scoring keyboard layouts.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

Calculate the total physical distance traveled by fingers when typing text with a specified keyboard layout. 
This approach assumes users have their fingers positioned above the home row.

Usage:

  # Basic usage
  python distance_scorer_new.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "the quick brown fox jumps over the lazy dog"

  # With text file
  python distance_scorer_new.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text-file sample.txt

  # CSV output
  python distance_scorer_new.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world" --csv

  # Score only
  python distance_scorer_new.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world" --score-only
"""

import sys
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
from pathlib import Path

# Import our new framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config
from framework.layout_utils import filter_to_letters_only
from framework.text_utils import extract_bigrams, clean_text_for_analysis, validate_text_input
from framework.output_utils import print_results
from framework.cli_utils import create_standard_parser, handle_common_errors, get_layout_from_args


# Physical keyboard layout definitions (same as original)
STAGGERED_POSITION_MAP = {
    # Top row (no stagger reference point)
    'q': (0, 0),    'w': (19, 0),   'e': (38, 0),   'r': (57, 0),   't': (76, 0),
    # Home row (staggered 5mm right from top row)
    'a': (5, 19),   's': (24, 19),  'd': (43, 19),  'f': (62, 19),  'g': (81, 19),
    # Bottom row (staggered 10mm right from home row)
    'z': (15, 38),  'x': (34, 38),  'c': (53, 38),  'v': (72, 38),  'b': (91, 38),
    # Top row continued
    'y': (95, 0),   'u': (114, 0),  'i': (133, 0),  'o': (152, 0),  'p': (171, 0),  '[': (190, 0),
    # Home row continued
    'h': (100, 19), 'j': (119, 19), 'k': (138, 19), 'l': (157, 19), ';': (176, 19), "'": (195, 19),
    # Bottom row continued
    'n': (110, 38), 'm': (129, 38), ',': (148, 38), '.': (167, 38), '/': (186, 38)
}

FINGER_MAP = {
    'q': 4, 'w': 3, 'e': 2, 'r': 1, 't': 1,
    'a': 4, 's': 3, 'd': 2, 'f': 1, 'g': 1,
    'z': 4, 'x': 3, 'c': 2, 'v': 1, 'b': 1,
    'y': 1, 'u': 1, 'i': 2, 'o': 3, 'p': 4,
    'h': 1, 'j': 1, 'k': 2, 'l': 3, ';': 4, 
    'n': 1, 'm': 1, ',': 2, '.': 3, '/': 4,
    '[': 4, "'": 4
}

COLUMN_MAP = {
    'q': 1, 'w': 2, 'e': 3, 'r': 4, 't': 5, 
    'a': 1, 's': 2, 'd': 3, 'f': 4, 'g': 5, 
    'z': 1, 'x': 2, 'c': 3, 'v': 4, 'b': 5,
    'y': 6, 'u': 7, 'i': 8, 'o': 9, 'p': 10, 
    'h': 6, 'j': 7, 'k': 8, 'l': 9, ';': 10, 
    'n': 6, 'm': 7, ',': 8, '.': 9, '/': 10,
    '[': 11, "'": 11
}


def calculate_euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions in mm."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return sqrt(dx * dx + dy * dy)


def same_hand(char1: str, char2: str) -> bool:
    """Check if two characters are typed by the same hand."""
    return (COLUMN_MAP[char1] < 6 and COLUMN_MAP[char2] < 6) or \
           (COLUMN_MAP[char1] > 5 and COLUMN_MAP[char2] > 5)


def same_finger(char1: str, char2: str) -> bool:
    """Check if two characters are typed by the same finger."""
    return same_hand(char1, char2) and FINGER_MAP[char1] == FINGER_MAP[char2]


class DistanceScorer(BaseLayoutScorer):
    """
    Distance-based keyboard layout scorer using the unified framework.
    
    Calculates the total physical distance traveled by fingers when typing text
    on a given keyboard layout.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the distance scorer.
        
        Args:
            layout_mapping: Dict mapping characters to QWERTY positions
            config: Optional configuration dictionary
        """
        super().__init__(layout_mapping, config)
        
        # Distance scorer doesn't need external data files
        self._data_loaded = True
        
        # Validate that all mapped positions exist in our position map
        self._validate_positions()
    
    def _validate_positions(self) -> None:
        """Validate that all layout positions exist in the physical position map."""
        issues = []
        quiet_mode = self.config.get('quiet_mode', False)
        
        for char, pos in self.layout_mapping.items():
            if pos.lower() not in STAGGERED_POSITION_MAP:
                issues.append(f"Position '{pos}' for character '{char}' not found in position map")
        
        if issues and not quiet_mode:
            print("Position validation warnings:")
            for issue in issues:
                print(f"  {issue}")
            print()
    
    def load_data_files(self) -> None:
        """
        Load required data files for scoring.
        
        Distance scorer uses built-in position data, so no external files needed.
        """
        # No external data files needed for distance scoring
        pass
    
    def get_physical_position(self, char: str) -> Optional[Tuple[float, float]]:
        """Get the physical position of a character based on the layout mapping."""
        char_upper = char.upper()
        if char_upper in self.layout_mapping:
            qwerty_pos = self.layout_mapping[char_upper].lower()
            return STAGGERED_POSITION_MAP.get(qwerty_pos)
        return None
    
    def calculate_bigram_distance(self, char1: str, char2: str) -> float:
        """Calculate the distance traveled for a single bigram."""
        pos1 = self.get_physical_position(char1)
        pos2 = self.get_physical_position(char2)
        
        if pos1 is None or pos2 is None:
            return 0.0  # Can't calculate distance for unmapped characters
        
        # Same character = no movement
        if char1 == char2:
            return 0.0
        
        return calculate_euclidean_distance(pos1, pos2)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and calculate comprehensive distance statistics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict containing various distance metrics and statistics
        """
        if not text.strip():
            return self._empty_results()
        
        # Use framework text utilities
        bigrams = extract_bigrams(text, respect_word_boundaries=True, normalize_case=True)
        
        if len(bigrams) == 0:
            return self._empty_results()
        
        # Calculate distances
        bigram_distances = []
        same_finger_distances = []
        same_hand_distances = []
        different_hand_distances = []
        
        total_distance = 0.0
        valid_bigrams = 0
        
        for char1, char2 in bigrams:
            distance = self.calculate_bigram_distance(char1, char2)
            
            if distance > 0:  # Only count valid distances
                bigram_distances.append(distance)
                total_distance += distance
                valid_bigrams += 1
                
                # Categorize by finger/hand usage
                if char1 in self.layout_mapping and char2 in self.layout_mapping:
                    qwerty_pos1 = self.layout_mapping[char1.upper()].lower()
                    qwerty_pos2 = self.layout_mapping[char2.upper()].lower()
                    
                    if qwerty_pos1 in FINGER_MAP and qwerty_pos2 in FINGER_MAP:
                        if same_finger(qwerty_pos1, qwerty_pos2):
                            same_finger_distances.append(distance)
                        elif same_hand(qwerty_pos1, qwerty_pos2):
                            same_hand_distances.append(distance)
                        else:
                            different_hand_distances.append(distance)
        
        # Calculate statistics
        if valid_bigrams > 0:
            avg_distance = total_distance / valid_bigrams
            max_distance = max(bigram_distances) if bigram_distances else 0.0
            min_distance = min(bigram_distances) if bigram_distances else 0.0
        else:
            avg_distance = max_distance = min_distance = 0.0
        
        # Calculate category averages
        avg_same_finger = sum(same_finger_distances) / len(same_finger_distances) if same_finger_distances else 0.0
        avg_same_hand = sum(same_hand_distances) / len(same_hand_distances) if same_hand_distances else 0.0
        avg_different_hand = sum(different_hand_distances) / len(different_hand_distances) if different_hand_distances else 0.0
        
        return {
            'total_distance': total_distance,
            'average_distance': avg_distance,
            'max_distance': max_distance,
            'min_distance': min_distance,
            'total_bigrams': len(bigrams),
            'valid_bigrams': valid_bigrams,
            'coverage': valid_bigrams / len(bigrams) if bigrams else 0.0,
            'avg_same_finger_distance': avg_same_finger,
            'avg_same_hand_distance': avg_same_hand,
            'avg_different_hand_distance': avg_different_hand,
            'same_finger_count': len(same_finger_distances),
            'same_hand_count': len(same_hand_distances),
            'different_hand_count': len(different_hand_distances),
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'total_distance': 0.0,
            'average_distance': 0.0,
            'max_distance': 0.0,
            'min_distance': 0.0,
            'total_bigrams': 0,
            'valid_bigrams': 0,
            'coverage': 0.0,
            'avg_same_finger_distance': 0.0,
            'avg_same_hand_distance': 0.0,
            'avg_different_hand_distance': 0.0,
            'same_finger_count': 0,
            'same_hand_count': 0,
            'different_hand_count': 0,
        }
    
    def calculate_scores(self) -> ScoreResult:
        """
        Calculate layout scores using distance analysis.
        
        Returns:
            ScoreResult containing primary score, components, and metadata
        """
        # Get text from config or raise error
        text = self.config.get('text', '')
        
        if not text:
            # Create empty result for missing text
            result = ScoreResult(
                primary_score=0.0,
                components={
                    'total_distance': 0.0,
                    'average_distance': 0.0,
                    'coverage': 0.0,
                },
                metadata={'error': 'No text provided for analysis'}
            )
            return result
        
        # Validate text input
        text_issues = validate_text_input(text)
        if text_issues:
            quiet_mode = self.config.get('quiet_mode', False)
            if not quiet_mode:
                print("Text validation warnings:")
                for issue in text_issues:
                    print(f"  {issue}")
                print()
        
        # Analyze the text
        analysis = self.analyze_text(text)
        
        # Calculate normalized score (same as original)
        avg_distance = analysis['average_distance']
        if avg_distance == 0:
            normalized_score = 0.0
        else:
            # Convert distance to score where lower distance = higher score
            normalized_score = 1000.0 / (avg_distance + 1000.0)
        
        # Create result using framework structure
        result = ScoreResult(
            primary_score=normalized_score,
            components={
                'total_distance': analysis['total_distance'],
                'average_distance': analysis['average_distance'],
                'max_distance': analysis['max_distance'],
                'min_distance': analysis['min_distance'],
                'coverage': analysis['coverage'],
            },
            metadata={
                'text_length': len(text),
                'distance_metric': self.config.get('scoring_options', {}).get('distance_metric', 'euclidean'),
            },
            validation_info={
                'total_bigrams': analysis['total_bigrams'],
                'valid_bigrams': analysis['valid_bigrams'],
                'coverage_percentage': analysis['coverage'] * 100,
            },
            detailed_breakdown={
                'finger_categories': {
                    'same_finger_avg': analysis['avg_same_finger_distance'],
                    'same_hand_avg': analysis['avg_same_hand_distance'],
                    'different_hand_avg': analysis['avg_different_hand_distance'],
                    'same_finger_count': analysis['same_finger_count'],
                    'same_hand_count': analysis['same_hand_count'],
                    'different_hand_count': analysis['different_hand_count'],
                },
            }
        )
        
        return result


@handle_common_errors
def main() -> int:
    """Main entry point using the standardized framework."""
    
    # Create standardized CLI parser
    cli_parser = create_standard_parser('distance_scorer')
    args = cli_parser.parse_args()
    
    try:
        # Load configuration
        config = load_scorer_config('distance_scorer', args.config)
        
        # Override with command-line arguments
        config['quiet_mode'] = args.quiet
        
        # Get text input
        if args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except FileNotFoundError:
                print(f"Error: Text file not found: {args.text_file}")
                return 1
        elif args.text:
            text = args.text
        else:
            print("Error: Must provide either --text or --text-file")
            return 1
        
        if not text.strip():
            print("Error: Empty text provided")
            return 1
        
        # Add text to config
        config['text'] = text
        
        # Get layout mapping from arguments
        letters, positions, layout_mapping = get_layout_from_args(args)
        
        # Filter to letters only (same as original behavior)
        layout_mapping = filter_to_letters_only(layout_mapping)
        
        if not layout_mapping:
            print("Error: No letters found in layout")
            return 1
        
        # Create and run scorer
        scorer = DistanceScorer(layout_mapping, config)
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