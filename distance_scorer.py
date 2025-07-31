#!/usr/bin/env python3
"""
Distance Layout Scorer for scoring keyboard layouts.

(c) Arno Klein (arnoklein.info), MIT License (see LICENSE)

Calculate the total physical distance traveled by fingers when typing text with a specified keyboard layout.
This approach correctly tracks cumulative finger travel by maintaining each finger's position:

  - **Initial state**: All fingers start at their home row positions
  - **When a finger types a key**: Calculate distance from that finger's current position to the new key position
  - **Update finger position**: Track where each finger is now located
  - **Next use of same finger**: Calculate distance from finger's last position to the new key

This provides an accurate measure of actual typing effort, unlike approaches that incorrectly
sum distances between all consecutive key positions regardless of which fingers type them.

Usage:

  # Basic usage
  python distance_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "the quick brown fox jumps over the lazy dog"

  # With cross-hand filtering
  python distance_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world" --ignore-cross-hand

  # With text file
  python distance_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text-file sample.txt

  # CSV output
  python distance_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world" --csv

  # Score only
  python distance_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world" --score-only
"""

import sys
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
from pathlib import Path

# Import framework components
from framework.base_scorer import BaseLayoutScorer, ScoreResult
from framework.config_loader import load_scorer_config
from framework.layout_utils import filter_to_letters_only, is_same_hand_pair
from framework.text_utils import extract_bigrams, clean_text_for_analysis, validate_text_input
from framework.output_utils import print_results
from framework.cli_utils import create_standard_parser, handle_common_errors, get_layout_from_args


# Physical keyboard layout definitions
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

# Home row positions for each finger (where fingers start)
HOME_ROW_POSITIONS = {
    'L4': 'a',  # Left pinky
    'L3': 's',  # Left ring
    'L2': 'd',  # Left middle
    'L1': 'f',  # Left index
    'R1': 'j',  # Right index
    'R2': 'k',  # Right middle
    'R3': 'l',  # Right ring
    'R4': ';'   # Right pinky
}


def calculate_euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions in mm."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return sqrt(dx * dx + dy * dy)


def get_finger_id(char: str) -> Optional[str]:
    """Get unique finger identifier for a character (combines hand and finger number)."""
    if char not in FINGER_MAP or char not in COLUMN_MAP:
        return None
    
    hand = 'L' if COLUMN_MAP[char] < 6 else 'R'
    finger_num = FINGER_MAP[char]
    return f"{hand}{finger_num}"


class DistanceScorer(BaseLayoutScorer):
    """
    Distance-based keyboard layout scorer that tracks cumulative finger travel.
    
    Tracks each finger's position and calculates cumulative distance traveled.
    All fingers start at home row positions. When a finger types a key,
    we calculate distance from its current position to the new key and update position.
    
    This provides a more accurate measure of actual typing effort compared to
    approaches which incorrectly sum all key-pair distances.
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
        
        # Initialize finger positions at home row
        self.finger_positions = {}
        self._initialize_finger_positions()
        
        # Cross-hand filtering option
        scoring_options = self.config.get('scoring_options', {})
        self.ignore_cross_hand = scoring_options.get('ignore_cross_hand', False)
        
        # Validate that all mapped positions exist in our position map
        self._validate_positions()
    
    def _initialize_finger_positions(self) -> None:
        """Initialize all fingers to their home row positions."""
        self.finger_positions = {}
        for finger_id, home_key in HOME_ROW_POSITIONS.items():
            self.finger_positions[finger_id] = home_key
    
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
        """Load required data files for scoring (none needed for distance scorer)."""
        pass
    
    def get_physical_position(self, qwerty_key: str) -> Optional[Tuple[float, float]]:
        """Get the physical position of a QWERTY key."""
        return STAGGERED_POSITION_MAP.get(qwerty_key.lower())
    
    def get_finger_for_char(self, char: str) -> Optional[str]:
        char_lower = char.lower()
        if char_lower in self.layout_mapping:
            qwerty_pos = self.layout_mapping[char_lower].lower()
            return get_finger_id(qwerty_pos)
        return None

    def get_qwerty_key_for_char(self, char: str) -> Optional[str]:
        char_lower = char.lower()
        if char_lower in self.layout_mapping:
            return self.layout_mapping[char_lower].lower()
        return None
    
    def calculate_finger_travel_distance(self, char: str) -> float:
        """
        Calculate the distance a finger travels to type this character.
        
        Args:
            char: The character being typed
            
        Returns:
            Distance the responsible finger travels from its current position
        """
        # Get the finger responsible for this character
        finger_id = self.get_finger_for_char(char)
        if finger_id is None:
            return 0.0
        
        # Get the QWERTY key this character maps to
        target_key = self.get_qwerty_key_for_char(char)
        if target_key is None:
            return 0.0
        
        # Get finger's current position
        current_key = self.finger_positions.get(finger_id)
        if current_key is None:
            return 0.0
        
        # Get physical positions
        current_pos = self.get_physical_position(current_key)
        target_pos = self.get_physical_position(target_key)
        
        if current_pos is None or target_pos is None:
            return 0.0
        
        # Calculate distance
        distance = calculate_euclidean_distance(current_pos, target_pos)
        
        # Update finger position
        self.finger_positions[finger_id] = target_key
        
        return distance
    
    def analyze_text_finger_travel(self, text: str) -> Dict[str, Any]:
        """
        Analyze finger travel distance for the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with travel analysis results
        """
        if not text.strip():
            return self._empty_results()
        
        # Reset finger positions to home row
        self._initialize_finger_positions()
        
        # Get all alphabetic characters in order
        chars = [c.lower() for c in text if c.isalpha() or c in ',.;\'/-=[]\\']
        
        if len(chars) == 0:
            return self._empty_results()
        
        # Filter cross-hand bigrams if requested
        if self.ignore_cross_hand:
            filtered_chars = []
            for i, char in enumerate(chars):
                if i == 0:
                    filtered_chars.append(char)
                    continue
                
                # Check if this forms a cross-hand bigram with previous character
                prev_char = chars[i - 1]
                prev_pos = self.get_qwerty_key_for_char(prev_char)
                curr_pos = self.get_qwerty_key_for_char(char)
                
                if prev_pos and curr_pos and not is_same_hand_pair(prev_pos, curr_pos):
                    # Skip this character (cross-hand bigram)
                    continue
                else:
                    filtered_chars.append(char)
            
            chars = filtered_chars
            
            if len(chars) == 0:
                return self._empty_results()
        
        # Calculate travel distance for each character
        travel_distances = []
        total_travel = 0.0
        
        # Track travel by finger
        finger_travels = {}
        finger_keystrokes = {}
        
        for char in chars:
            distance = self.calculate_finger_travel_distance(char)
            
            if distance >= 0:  # Include zero distances (staying on same key)
                travel_distances.append(distance)
                total_travel += distance
                
                # Track by finger
                finger = self.get_finger_for_char(char)
                if finger:
                    if finger not in finger_travels:
                        finger_travels[finger] = 0.0
                        finger_keystrokes[finger] = 0
                    finger_travels[finger] += distance
                    finger_keystrokes[finger] += 1
        
        # Calculate statistics
        keystroke_count = len(travel_distances)
        if keystroke_count > 0:
            avg_travel = total_travel / keystroke_count
            max_travel = max(travel_distances) if travel_distances else 0.0
            min_travel = min(travel_distances) if travel_distances else 0.0
        else:
            avg_travel = max_travel = min_travel = 0.0
        
        # Calculate per-finger statistics
        finger_stats = {}
        for finger in finger_travels:
            finger_stats[finger] = {
                'total_travel': finger_travels[finger],
                'keystroke_count': finger_keystrokes[finger],
                'avg_travel': finger_travels[finger] / finger_keystrokes[finger] if finger_keystrokes[finger] > 0 else 0.0
            }
        
        return {
            'total_travel': total_travel,
            'average_travel': avg_travel,
            'max_travel': max_travel,
            'min_travel': min_travel,
            'keystroke_count': keystroke_count,
            'finger_statistics': finger_stats,
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'total_travel': 0.0,
            'average_travel': 0.0,
            'max_travel': 0.0,
            'min_travel': 0.0,
            'keystroke_count': 0,
            'finger_statistics': {},
        }
    
    def calculate_scores(self) -> ScoreResult:
        """
        Calculate layout scores using cumulative finger travel analysis.
        
        Returns:
            ScoreResult containing primary score, components, and metadata
        """
        # Get text from config
        text = self.config.get('text', '')
        
        if not text:
            result = ScoreResult(
                primary_score=0.0,
                components={
                    'total_travel': 0.0,
                    'average_travel': 0.0,
                    'keystroke_count': 0,
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
        
        # Analyze finger travel
        analysis = self.analyze_text_finger_travel(text)
        
        # Calculate normalized score (lower travel = higher score)
        avg_travel = analysis['average_travel']
        if avg_travel == 0:
            normalized_score = 1.0  # Perfect score for no travel
        else:
            # Normalize score where lower average travel = higher score
            normalized_score = 1000.0 / (avg_travel + 1000.0)
        
        # Create result using framework structure
        result = ScoreResult(
            primary_score=normalized_score,
            components={
                'total_travel': analysis['total_travel'],
                'average_travel': analysis['average_travel'],
                'max_travel': analysis['max_travel'],
                'min_travel': analysis['min_travel'],
                'keystroke_count': analysis['keystroke_count'],
            },
            metadata={
                'text_length': len(text),
                'scoring_method': 'cumulative_finger_travel',
                'ignore_cross_hand': self.ignore_cross_hand,
                'description': 'Tracks each finger position and calculates cumulative travel distance with optional cross-hand filtering',
            },
            validation_info={
                'keystroke_count': analysis['keystroke_count'],
                'text_characters': len([c for c in text if c.isalpha() or c in ',.;\'/-=[]\\'])],
                'cross_hand_filtering': self.ignore_cross_hand,
            },
            detailed_breakdown={
                'finger_statistics': analysis['finger_statistics'],
                'travel_summary': {
                    'total_distance_mm': analysis['total_travel'],
                    'average_per_keystroke_mm': analysis['average_travel'],
                    'keystrokes_analyzed': analysis['keystroke_count']
                }
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
        
        # Handle cross-hand filtering
        if hasattr(args, 'ignore_cross_hand') and args.ignore_cross_hand:
            if 'scoring_options' not in config:
                config['scoring_options'] = {}
            config['scoring_options']['ignore_cross_hand'] = True
        
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