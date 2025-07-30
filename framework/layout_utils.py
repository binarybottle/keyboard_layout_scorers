#!/usr/bin/env python3
"""
Layout utilities for keyboard layout scoring.

Common functions for creating, validating, and manipulating layout mappings.
"""

from typing import Dict, List, Tuple, Optional, Set
import re


def create_layout_mapping(letters: str, positions: str) -> Dict[str, str]:
    """
    Create a layout mapping from letter and position strings.
    
    Args:
        letters: String of characters (e.g., 'etaoinshrlcu')
        positions: String of corresponding positions (e.g., 'FDESGJWXRTYZ')
        
    Returns:
        Dict mapping characters to positions (lowercase chars to uppercase positions)
        
    Raises:
        ValueError: If strings have different lengths
    """
    if len(letters) != len(positions):
        raise ValueError(f"Letters length ({len(letters)}) != positions length ({len(positions)})")
    
    # Convert to consistent case
    letters_clean = letters.lower()
    positions_clean = positions.upper()
    
    return dict(zip(letters_clean, positions_clean))


def validate_layout_mapping(mapping: Dict[str, str], 
                          strict: bool = True,
                          allowed_chars: Optional[Set[str]] = None) -> List[str]:
    """
    Validate a layout mapping for correctness and consistency.
    
    Args:
        mapping: Character to position mapping
        strict: If True, apply strict validation rules
        allowed_chars: Set of allowed characters (None = no restriction)
        
    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []
    
    if not mapping:
        issues.append("Layout mapping is empty")
        return issues
    
    # Check for empty keys or values
    for char, pos in mapping.items():
        if not char:
            issues.append("Empty character found in mapping")
        if not pos:
            issues.append(f"Empty position for character '{char}'")
        
        # Check character restrictions
        if allowed_chars is not None and char not in allowed_chars:
            issues.append(f"Character '{char}' not in allowed set")
    
    # Check for duplicate positions
    positions = list(mapping.values())
    position_counts = {}
    for pos in positions:
        position_counts[pos] = position_counts.get(pos, 0) + 1
    
    duplicates = [pos for pos, count in position_counts.items() if count > 1]
    if duplicates:
        issues.append(f"Duplicate positions: {duplicates}")
    
    if strict:
        # Strict validation rules
        
        # Check character case consistency (should be lowercase)
        non_lowercase = [char for char in mapping.keys() if char != char.lower()]
        if non_lowercase:
            issues.append(f"Characters should be lowercase: {non_lowercase}")
        
        # Check position case consistency (should be uppercase)
        non_uppercase = [pos for pos in mapping.values() if pos != pos.upper()]
        if non_uppercase:
            issues.append(f"Positions should be uppercase: {non_uppercase}")
        
        # Check for reasonable character set (alphanumeric + common punctuation)
        valid_chars = set('abcdefghijklmnopqrstuvwxyz0123456789.,;\'')
        invalid_chars = [char for char in mapping.keys() if char not in valid_chars]
        if invalid_chars:
            issues.append(f"Unusual characters found: {invalid_chars}")
    
    return issues


def filter_to_letters_only(mapping: Dict[str, str], 
                          preserve_numbers: bool = False) -> Dict[str, str]:
    """
    Filter layout mapping to contain only alphabetic characters.
    
    Args:
        mapping: Original character to position mapping
        preserve_numbers: If True, also keep numeric characters
        
    Returns:
        Filtered mapping containing only letters (and optionally numbers)
    """
    filtered = {}
    
    for char, pos in mapping.items():
        if char.isalpha():
            filtered[char] = pos
        elif preserve_numbers and char.isdigit():
            filtered[char] = pos
    
    return filtered


def get_layout_statistics(mapping: Dict[str, str]) -> Dict[str, any]:
    """
    Get statistics about a layout mapping.
    
    Args:
        mapping: Character to position mapping
        
    Returns:
        Dictionary with layout statistics
    """
    if not mapping:
        return {
            'total_chars': 0,
            'letters': 0,
            'numbers': 0,
            'punctuation': 0,
            'coverage': 0.0
        }
    
    stats = {
        'total_chars': len(mapping),
        'letters': sum(1 for char in mapping.keys() if char.isalpha()),
        'numbers': sum(1 for char in mapping.keys() if char.isdigit()),
        'punctuation': sum(1 for char in mapping.keys() if not char.isalnum()),
        'unique_positions': len(set(mapping.values())),
        'chars_list': sorted(mapping.keys()),
        'positions_list': sorted(set(mapping.values())),
    }
    
    # Calculate coverage of standard alphabet
    standard_letters = set('abcdefghijklmnopqrstuvwxyz')
    mapped_letters = set(char.lower() for char in mapping.keys() if char.isalpha())
    stats['alphabet_coverage'] = len(mapped_letters & standard_letters) / len(standard_letters)
    
    return stats


def normalize_layout_strings(letters: str, positions: str) -> Tuple[str, str]:
    """
    Normalize and clean layout strings for consistent processing.
    
    Args:
        letters: String of characters
        positions: String of positions
        
    Returns:
        Tuple of (cleaned_letters, cleaned_positions)
    """
    # Remove whitespace and convert to consistent case
    letters_clean = re.sub(r'\s+', '', letters.lower())
    positions_clean = re.sub(r'\s+', '', positions.upper())
    
    return letters_clean, positions_clean


def extract_letter_pairs_only(mapping: Dict[str, str], 
                             letters: str, 
                             positions: str) -> Tuple[str, str]:
    """
    Extract only letter-position pairs from the input strings.
    
    Args:
        mapping: Full character to position mapping
        letters: Original letters string
        positions: Original positions string
        
    Returns:
        Tuple of (letters_only, positions_only) with non-letters filtered out
    """
    filtered_letters = []
    filtered_positions = []
    
    for char, pos in zip(letters, positions):
        if char.isalpha():
            filtered_letters.append(char.lower())
            filtered_positions.append(pos.upper())
    
    return ''.join(filtered_letters), ''.join(filtered_positions)


def get_hand_mapping() -> Dict[str, str]:
    """
    Get standard hand assignments for QWERTY positions.
    
    Returns:
        Dict mapping position keys to 'L' (left) or 'R' (right)
    """
    return {
        # Left hand positions
        '1': 'L', '2': 'L', '3': 'L', '4': 'L', '5': 'L',
        'Q': 'L', 'W': 'L', 'E': 'L', 'R': 'L', 'T': 'L',
        'A': 'L', 'S': 'L', 'D': 'L', 'F': 'L', 'G': 'L',
        'Z': 'L', 'X': 'L', 'C': 'L', 'V': 'L', 'B': 'L',
        
        # Right hand positions
        '6': 'R', '7': 'R', '8': 'R', '9': 'R', '0': 'R',
        'Y': 'R', 'U': 'R', 'I': 'R', 'O': 'R', 'P': 'R',
        'H': 'R', 'J': 'R', 'K': 'R', 'L': 'R', ';': 'R',
        'N': 'R', 'M': 'R', ',': 'R', '.': 'R', '/': 'R',
        "'": 'R', '[': 'R', ']': 'R',
    }


def is_same_hand_pair(pos1: str, pos2: str, hand_mapping: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if two positions are typed by the same hand.
    
    Args:
        pos1: First position
        pos2: Second position
        hand_mapping: Custom hand mapping (uses standard QWERTY if None)
        
    Returns:
        True if same hand, False otherwise
    """
    if hand_mapping is None:
        hand_mapping = get_hand_mapping()
    
    pos1_upper = pos1.upper()
    pos2_upper = pos2.upper()
    
    if pos1_upper not in hand_mapping or pos2_upper not in hand_mapping:
        return False
    
    return hand_mapping[pos1_upper] == hand_mapping[pos2_upper]


def analyze_hand_distribution(mapping: Dict[str, str]) -> Dict[str, any]:
    """
    Analyze the hand distribution of a layout mapping.
    
    Args:
        mapping: Character to position mapping
        
    Returns:
        Dictionary with hand distribution statistics
    """
    hand_mapping = get_hand_mapping()
    
    left_chars = []
    right_chars = []
    unknown_chars = []
    
    for char, pos in mapping.items():
        pos_upper = pos.upper()
        hand = hand_mapping.get(pos_upper)
        
        if hand == 'L':
            left_chars.append(char)
        elif hand == 'R':
            right_chars.append(char)
        else:
            unknown_chars.append(char)
    
    total_chars = len(mapping)
    
    return {
        'left_hand_chars': sorted(left_chars),
        'right_hand_chars': sorted(right_chars),
        'unknown_hand_chars': sorted(unknown_chars),
        'left_count': len(left_chars),
        'right_count': len(right_chars),
        'unknown_count': len(unknown_chars),
        'left_percentage': len(left_chars) / total_chars * 100 if total_chars > 0 else 0,
        'right_percentage': len(right_chars) / total_chars * 100 if total_chars > 0 else 0,
        'balance_ratio': min(len(left_chars), len(right_chars)) / max(len(left_chars), len(right_chars)) if max(len(left_chars), len(right_chars)) > 0 else 0,
    }


def compare_layouts(mapping1: Dict[str, str], 
                   mapping2: Dict[str, str], 
                   name1: str = "Layout 1", 
                   name2: str = "Layout 2") -> Dict[str, any]:
    """
    Compare two layout mappings and return differences.
    
    Args:
        mapping1: First layout mapping
        mapping2: Second layout mapping
        name1: Name for first layout
        name2: Name for second layout
        
    Returns:
        Dictionary with comparison results
    """
    chars1 = set(mapping1.keys())
    chars2 = set(mapping2.keys())
    
    common_chars = chars1 & chars2
    only_in_1 = chars1 - chars2
    only_in_2 = chars2 - chars1
    
    # Find position differences for common characters
    position_diffs = []
    for char in common_chars:
        if mapping1[char] != mapping2[char]:
            position_diffs.append((char, mapping1[char], mapping2[char]))
    
    return {
        'common_chars': sorted(common_chars),
        'only_in_first': sorted(only_in_1),
        'only_in_second': sorted(only_in_2),
        'position_differences': position_diffs,
        'similarity_ratio': len(common_chars) / len(chars1 | chars2) if (chars1 | chars2) else 1.0,
        'identical_positions': len(common_chars) - len(position_diffs),
        'total_differences': len(position_diffs) + len(only_in_1) + len(only_in_2),
    }