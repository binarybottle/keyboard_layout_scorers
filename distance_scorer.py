#!/usr/bin/env python3
"""
Distance-based keyboard layout scorer.

This script calculates the total physical distance traveled by fingers when typing text
on different keyboard layouts. It assumes users have their fingers positioned above
the home row and measures the Euclidean distance between key positions for each
character transition.

Features:
- Uses physical staggered keyboard layout positions (19mm spacing)
- Calculates per-bigram distance and total distance
- Supports different layout mappings
- Provides distance statistics and normalized scores
- Can output detailed per-bigram analysis

Usage:
    # Basic distance scoring
    python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text "the text"
    
    # With text file input
    python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text-file "sample.txt"
    
    # CSV output format
    python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text "the text" --csv
    
    # Detailed bigram analysis
    python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text "the text" --detailed

Example layouts:
    qwerty_layout = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"
    dvorak_layout = "',.PYFGCRLAOEUIDHTNS;QJKXBMWVZ"
"""

import argparse
import sys
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from math import sqrt

# Keyboard layout definitions
# from https://github.com/binarybottle/typing_preferences_to_comfort_scores/tree/main
staggered_position_map = {
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

finger_map = {
    'q': 4, 'w': 3, 'e': 2, 'r': 1, 't': 1,
    'a': 4, 's': 3, 'd': 2, 'f': 1, 'g': 1,
    'z': 4, 'x': 3, 'c': 2, 'v': 1, 'b': 1,
    'y': 1, 'u': 1, 'i': 2, 'o': 3, 'p': 4,
    'h': 1, 'j': 1, 'k': 2, 'l': 3, ';': 4, 
    'n': 1, 'm': 1, ',': 2, '.': 3, '/': 4,
    '[': 4, "'": 4
}

column_map = {
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
    return (column_map[char1] < 6 and column_map[char2] < 6) or \
           (column_map[char1] > 5 and column_map[char2] > 5)


def same_finger(char1: str, char2: str) -> bool:
    """Check if two characters are typed by the same finger."""
    return same_hand(char1, char2) and finger_map[char1] == finger_map[char2]


class DistanceScorer:
    """
    Distance-based keyboard layout scorer.
    
    Calculates the total physical distance traveled by fingers when typing text
    on a given keyboard layout.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], csv_mode: bool = False):
        """
        Initialize the distance scorer.
        
        Args:
            layout_mapping: Dict mapping characters to QWERTY positions (e.g., {'a': 'F', 'b': 'D'})
            csv_mode: If True, suppress warning messages for clean CSV output
        """
        # Convert to uppercase for consistency
        self.layout_mapping = {char.upper(): pos.upper() for char, pos in layout_mapping.items()}
        self.csv_mode = csv_mode
        
        # Create reverse mapping for quick lookup
        self.position_to_char = {pos.upper(): char for char, pos in self.layout_mapping.items()}
        
        # Validate that all mapped positions exist in our position map
        if not csv_mode:  # Only show warnings in non-CSV mode
            for char, pos in self.layout_mapping.items():
                if pos.lower() not in staggered_position_map:
                    print(f"Warning: Position '{pos}' for character '{char}' not found in position map")
    
    def get_physical_position(self, char: str) -> Optional[Tuple[float, float]]:
        """Get the physical position of a character based on the layout mapping."""
        char_upper = char.upper()
        if char_upper in self.layout_mapping:
            qwerty_pos = self.layout_mapping[char_upper].lower()
            return staggered_position_map.get(qwerty_pos)
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean text by replacing non-alphabetic characters with spaces and converting to uppercase."""
        # Replace non-alphabetic characters with spaces to avoid false bigrams
        cleaned = re.sub(r'[^a-zA-Z]', ' ', text.upper())
        # Normalize multiple spaces to single spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def extract_bigrams(self, text: str) -> List[Tuple[str, str]]:
        """Extract all consecutive character pairs from text, respecting word boundaries."""
        cleaned_text = self.clean_text(text)
        bigrams = []
        
        # Split into words to avoid cross-word bigrams
        words = cleaned_text.split()
        
        for word in words:
            # Extract bigrams within each word
            for i in range(len(word) - 1):
                char1, char2 = word[i], word[i + 1]
                bigrams.append((char1, char2))
        
        return bigrams
    
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
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text and calculate comprehensive distance statistics.
        
        Returns:
            Dict containing various distance metrics and statistics
        """
        if not text.strip():
            return self._empty_results()
        
        bigrams = self.extract_bigrams(text)
        
        if len(bigrams) == 0:
            return self._empty_results()
        
        # Calculate distances
        bigram_distances = []
        same_finger_distances = []
        same_hand_distances = []
        different_hand_distances = []
        bigram_counts = Counter()
        
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
                    qwerty_pos1 = self.layout_mapping[char1].lower()
                    qwerty_pos2 = self.layout_mapping[char2].lower()
                    
                    if same_finger(qwerty_pos1, qwerty_pos2):
                        same_finger_distances.append(distance)
                    elif same_hand(qwerty_pos1, qwerty_pos2):
                        same_hand_distances.append(distance)
                    else:
                        different_hand_distances.append(distance)
            
            # Count bigram frequency regardless of distance calculation
            bigram_counts[(char1, char2)] += 1
        
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
        
        # Find most expensive bigrams
        bigram_distance_map = {}
        for char1, char2 in bigrams:
            bigram_key = (char1, char2)
            if bigram_key not in bigram_distance_map:
                distance = self.calculate_bigram_distance(char1, char2)
                if distance > 0:
                    bigram_distance_map[bigram_key] = distance
        
        # Sort by distance * frequency to find most impactful bigrams
        bigram_impact = []
        for (char1, char2), count in bigram_counts.items():
            distance = bigram_distance_map.get((char1, char2), 0.0)
            impact = distance * count
            if impact > 0:
                bigram_impact.append(((char1, char2), distance, count, impact))
        
        bigram_impact.sort(key=lambda x: x[3], reverse=True)  # Sort by impact
        
        return {
            'total_distance': total_distance,
            'average_distance': avg_distance,
            'max_distance': max_distance,
            'min_distance': min_distance,
            'total_bigrams': len(bigrams),
            'valid_bigrams': valid_bigrams,
            'coverage': valid_bigrams / len(bigrams) if bigrams else 0.0,
            'unique_bigrams': len(bigram_counts),
            'avg_same_finger_distance': avg_same_finger,
            'avg_same_hand_distance': avg_same_hand,
            'avg_different_hand_distance': avg_different_hand,
            'same_finger_count': len(same_finger_distances),
            'same_hand_count': len(same_hand_distances),
            'different_hand_count': len(different_hand_distances),
            'top_bigrams_by_impact': bigram_impact[:10],  # Top 10 most impactful
            'bigram_distances': bigram_distances,
            'bigram_counts': bigram_counts
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'total_distance': 0.0,
            'average_distance': 0.0,
            'max_distance': 0.0,
            'min_distance': 0.0,
            'total_bigrams': 0,
            'valid_bigrams': 0,
            'coverage': 0.0,
            'unique_bigrams': 0,
            'avg_same_finger_distance': 0.0,
            'avg_same_hand_distance': 0.0,
            'avg_different_hand_distance': 0.0,
            'same_finger_count': 0,
            'same_hand_count': 0,
            'different_hand_count': 0,
            'top_bigrams_by_impact': [],
            'bigram_distances': [],
            'bigram_counts': Counter()
        }
    
    def score_layout(self, text: str) -> float:
        """
        Calculate a normalized distance score for the layout.
        
        Lower distances are better, so we return the inverse of average distance.
        Score is normalized to a 0-1 scale where higher is better.
        
        Returns:
            float: Normalized score where higher values indicate better (lower distance) layouts
        """
        results = self.analyze_text(text)
        avg_distance = results['average_distance']
        
        if avg_distance == 0:
            return 0.0
        
        # Normalize: convert distance to score where lower distance = higher score
        # Using inverse relationship: score = 1000 / (avg_distance + 1000)
        # This gives scores roughly in 0-1 range, with lower distances getting higher scores
        normalized_score = 1000.0 / (avg_distance + 1000.0)
        
        return normalized_score


def print_results(results: Dict, detailed: bool = False) -> None:
    """Print formatted results from distance analysis."""
    print("\nDistance-based Layout Analysis")
    print("=" * 50)
    
    print(f"Total distance traveled:    {results['total_distance']:8.1f} mm")
    print(f"Average distance per bigram: {results['average_distance']:8.1f} mm")
    print(f"Maximum bigram distance:    {results['max_distance']:8.1f} mm")
    print(f"Minimum bigram distance:    {results['min_distance']:8.1f} mm")
    
    print(f"\nBigram statistics:")
    print(f"Total bigrams processed:    {results['total_bigrams']:8d}")
    print(f"Valid bigrams (calculable): {results['valid_bigrams']:8d}")
    print(f"Coverage:                   {results['coverage']*100:8.1f}%")
    print(f"Unique bigrams:             {results['unique_bigrams']:8d}")
    
    print(f"\nDistance by category:")
    print(f"Same finger average:        {results['avg_same_finger_distance']:8.1f} mm ({results['same_finger_count']} bigrams)")
    print(f"Same hand average:          {results['avg_same_hand_distance']:8.1f} mm ({results['same_hand_count']} bigrams)")
    print(f"Different hand average:     {results['avg_different_hand_distance']:8.1f} mm ({results['different_hand_count']} bigrams)")
    
    # Calculate normalized score
    avg_distance = results['average_distance']
    if avg_distance > 0:
        normalized_score = 1000.0 / (avg_distance + 1000.0)
        print(f"\nNormalized score (0-1):     {normalized_score:8.6f}  (higher = better)")
    
    if detailed and results['top_bigrams_by_impact']:
        print(f"\nTop 10 bigrams by impact (distance × frequency):")
        print(f"{'Bigram':<8} {'Distance':<10} {'Count':<8} {'Impact':<10}")
        print("-" * 40)
        for (char1, char2), distance, count, impact in results['top_bigrams_by_impact'][:10]:
            print(f"{char1}{char2:<6} {distance:8.1f} mm {count:8d} {impact:8.1f}")


def output_csv_error(error_msg: str) -> None:
    """Output error in CSV format for consistency."""
    print("error,message")
    print(f"error,\"{error_msg}\"")


def main() -> None:
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Calculate distance-based scores for keyboard layouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic distance scoring
  python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text "the quick brown fox"
  
  # From text file
  python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text-file "sample.txt"
  
  # CSV output
  python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text "the text" --csv
  
  # Simple score output (just the normalized score)
  python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text "the text" --score-only

  # Detailed analysis
  python distance_scorer.py --letters "ETAOIN" --qwerty-keys "FDESRJ" --text "the text" --detailed
        """
    )
    
    parser.add_argument("--letters", required=True,
                       help="String of characters in the layout (e.g., 'etaoinshrlcu')")
    parser.add_argument("--qwerty-keys", required=True,
                       help="String of corresponding QWERTY positions (e.g., 'FDESGJWXRTYZ')")
    parser.add_argument("--text",
                       help="Text to analyze (alternative to --text-file)")
    parser.add_argument("--text-file",
                       help="Path to text file to analyze (alternative to --text)")
    parser.add_argument("--csv", action="store_true",
                       help="Output in CSV format")
    parser.add_argument("--score-only", action="store_true",
                       help="Output only the normalized score")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed bigram analysis")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if len(args.letters) != len(args.qwerty_keys):
            error_msg = f"Character count ({len(args.letters)}) != Position count ({len(args.qwerty_keys)})"
            if args.csv:
                output_csv_error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return 1
        
        # Get text input
        if args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except FileNotFoundError:
                error_msg = f"Text file not found: {args.text_file}"
                if args.csv:
                    output_csv_error(error_msg)
                else:
                    print(f"Error: {error_msg}")
                return 1
        elif args.text:
            text = args.text
        else:
            error_msg = "Must provide either --text or --text-file"
            if args.csv:
                output_csv_error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return 1
        
        if not text.strip():
            error_msg = "Empty text provided"
            if args.csv:
                output_csv_error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return 1
        
        # Filter to only letters, keeping corresponding positions
        letter_pairs = [(char, pos) for char, pos in zip(args.letters, args.qwerty_keys) if char.isalpha()]
        
        if not letter_pairs:
            error_msg = "No letters found in --letters"
            if args.csv:
                output_csv_error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return 1
        
        # Create layout mapping
        filtered_letters = ''.join(pair[0] for pair in letter_pairs)
        filtered_positions = ''.join(pair[1] for pair in letter_pairs)
        layout_mapping = dict(zip(filtered_letters.upper(), filtered_positions.upper()))
        
        # Show layout info only in non-CSV mode
        if not args.csv and not args.score_only:
            print(f"Layout: {filtered_letters} → {filtered_positions}")
            print(f"Text length: {len(text):,} characters")
        
        # Calculate distance scores (pass csv_mode flag to suppress warnings)
        scorer = DistanceScorer(layout_mapping, csv_mode=args.csv)
        results = scorer.analyze_text(text)
        
        if args.score_only:
            # Output just the normalized score
            normalized_score = scorer.score_layout(text)
            print(f"{normalized_score:.6f}")
        
        elif args.csv:
            # CSV output - ONLY CSV data
            print("metric,value")
            print(f"total_distance,{results['total_distance']:.6f}")
            print(f"average_distance,{results['average_distance']:.6f}")
            print(f"max_distance,{results['max_distance']:.6f}")
            print(f"min_distance,{results['min_distance']:.6f}")
            print(f"total_bigrams,{results['total_bigrams']}")
            print(f"valid_bigrams,{results['valid_bigrams']}")
            print(f"coverage,{results['coverage']:.6f}")
            print(f"unique_bigrams,{results['unique_bigrams']}")
            print(f"avg_same_finger_distance,{results['avg_same_finger_distance']:.6f}")
            print(f"avg_same_hand_distance,{results['avg_same_hand_distance']:.6f}")
            print(f"avg_different_hand_distance,{results['avg_different_hand_distance']:.6f}")
            print(f"same_finger_count,{results['same_finger_count']}")
            print(f"same_hand_count,{results['same_hand_count']}")
            print(f"different_hand_count,{results['different_hand_count']}")
            
            # Add normalized score
            normalized_score = scorer.score_layout(text)
            print(f"normalized_score,{normalized_score:.6f}")
        
        else:
            # Human-readable output
            print_results(results, args.detailed)
        
    except Exception as e:
        error_msg = str(e)
        if args.csv:
            output_csv_error(error_msg)
        else:
            print(f"Error: {error_msg}")
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())