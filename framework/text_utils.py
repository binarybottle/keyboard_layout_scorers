#!/usr/bin/env python3
"""
Text utilities for keyboard layout scoring.

Common functions for processing text input, extracting bigrams, and analyzing text patterns.
"""

import re
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter, defaultdict


def clean_text_for_analysis(text: str, 
                           preserve_case: bool = False,
                           preserve_numbers: bool = False,
                           preserve_punctuation: bool = False) -> str:
    """
    Clean text for keyboard layout analysis.
    
    Args:
        text: Input text to clean
        preserve_case: If True, maintain original case
        preserve_numbers: If True, keep numeric characters
        preserve_punctuation: If True, keep punctuation characters
        
    Returns:
        Cleaned text suitable for analysis
    """
    if not text:
        return ""
    
    # Start with the original text
    cleaned = text
    
    # Handle case
    if not preserve_case:
        cleaned = cleaned.lower()
    
    # Define character sets to keep
    keep_chars = set('abcdefghijklmnopqrstuvwxyz')
    if preserve_case:
        keep_chars.update('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    if preserve_numbers:
        keep_chars.update('0123456789')
    if preserve_punctuation:
        keep_chars.update('.,;\'":!?-()[]{}')
    
    # Add space to keep for word boundaries
    keep_chars.add(' ')
    
    # Filter characters
    filtered_chars = []
    for char in cleaned:
        if char in keep_chars:
            filtered_chars.append(char)
        else:
            # Replace non-kept characters with space to maintain word boundaries
            if filtered_chars and filtered_chars[-1] != ' ':
                filtered_chars.append(' ')
    
    # Join and normalize whitespace
    result = ''.join(filtered_chars)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def extract_bigrams(text: str, 
                   respect_word_boundaries: bool = True,
                   filter_same_char: bool = False,
                   normalize_case: bool = True) -> List[Tuple[str, str]]:
    """
    Extract consecutive character pairs (bigrams) from text.
    
    Args:
        text: Input text to analyze
        respect_word_boundaries: If True, don't create bigrams across word boundaries
        filter_same_char: If True, exclude bigrams with same character (e.g., 'aa')
        normalize_case: If True, convert to lowercase
        
    Returns:
        List of character pair tuples
    """
    if not text:
        return []
    
    # Clean the text
    cleaned_text = clean_text_for_analysis(text, preserve_case=not normalize_case)
    
    bigrams = []
    
    if respect_word_boundaries:
        # Split into words and extract bigrams within each word
        words = cleaned_text.split()
        
        for word in words:
            if len(word) < 2:
                continue
                
            for i in range(len(word) - 1):
                char1, char2 = word[i], word[i + 1]
                
                # Apply filters
                if filter_same_char and char1 == char2:
                    continue
                
                bigrams.append((char1, char2))
    else:
        # Extract bigrams across the entire text (ignoring word boundaries)
        # Remove spaces first
        text_no_spaces = cleaned_text.replace(' ', '')
        
        for i in range(len(text_no_spaces) - 1):
            char1, char2 = text_no_spaces[i], text_no_spaces[i + 1]
            
            # Apply filters
            if filter_same_char and char1 == char2:
                continue
            
            bigrams.append((char1, char2))
    
    return bigrams


def get_character_frequencies(text: str, 
                            normalize: bool = True,
                            case_sensitive: bool = False) -> Dict[str, float]:
    """
    Calculate character frequencies in text.
    
    Args:
        text: Input text to analyze
        normalize: If True, return frequencies as proportions (sum to 1)
        case_sensitive: If True, treat uppercase and lowercase as different
        
    Returns:
        Dictionary mapping characters to their frequencies
    """
    if not text:
        return {}
    
    # Clean text
    cleaned_text = clean_text_for_analysis(text, preserve_case=case_sensitive)
    
    # Remove spaces for character frequency analysis
    chars_only = cleaned_text.replace(' ', '')
    
    if not chars_only:
        return {}
    
    # Count characters
    char_counts = Counter(chars_only)
    
    if normalize:
        total_chars = len(chars_only)
        return {char: count / total_chars for char, count in char_counts.items()}
    else:
        return dict(char_counts)


def get_bigram_frequencies(text: str, 
                         normalize: bool = True,
                         respect_word_boundaries: bool = True,
                         case_sensitive: bool = False) -> Dict[Tuple[str, str], float]:
    """
    Calculate bigram frequencies in text.
    
    Args:
        text: Input text to analyze
        normalize: If True, return frequencies as proportions (sum to 1)
        respect_word_boundaries: If True, don't create bigrams across word boundaries
        case_sensitive: If True, treat uppercase and lowercase as different
        
    Returns:
        Dictionary mapping bigram tuples to their frequencies
    """
    bigrams = extract_bigrams(text, 
                            respect_word_boundaries=respect_word_boundaries,
                            normalize_case=not case_sensitive)
    
    if not bigrams:
        return {}
    
    # Count bigrams
    bigram_counts = Counter(bigrams)
    
    if normalize:
        total_bigrams = len(bigrams)
        return {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}
    else:
        return {bigram: float(count) for bigram, count in bigram_counts.items()}


def analyze_text_patterns(text: str) -> Dict[str, any]:
    """
    Analyze various patterns in text for layout scoring insights.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with pattern analysis results
    """
    if not text:
        return {
            'total_chars': 0,
            'total_words': 0,
            'total_bigrams': 0,
            'unique_chars': 0,
            'unique_bigrams': 0,
            'avg_word_length': 0.0,
            'char_frequencies': {},
            'bigram_frequencies': {},
            'most_common_chars': [],
            'most_common_bigrams': [],
        }
    
    # Clean text
    cleaned_text = clean_text_for_analysis(text)
    
    # Basic statistics
    words = cleaned_text.split()
    chars_only = cleaned_text.replace(' ', '')
    bigrams = extract_bigrams(text)
    
    # Character analysis
    char_frequencies = get_character_frequencies(text)
    char_counter = Counter(chars_only)
    
    # Bigram analysis
    bigram_frequencies = get_bigram_frequencies(text)
    bigram_counter = Counter(bigrams)
    
    # Calculate statistics
    total_chars = len(chars_only)
    total_words = len(words)
    total_bigrams = len(bigrams)
    unique_chars = len(set(chars_only))
    unique_bigrams = len(set(bigrams))
    
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0.0
    
    # Most common items
    most_common_chars = char_counter.most_common(10)
    most_common_bigrams = bigram_counter.most_common(10)
    
    return {
        'total_chars': total_chars,
        'total_words': total_words,
        'total_bigrams': total_bigrams,
        'unique_chars': unique_chars,
        'unique_bigrams': unique_bigrams,
        'avg_word_length': avg_word_length,
        'char_frequencies': char_frequencies,
        'bigram_frequencies': bigram_frequencies,
        'most_common_chars': most_common_chars,
        'most_common_bigrams': most_common_bigrams,
        'vocabulary_diversity': unique_chars / total_chars if total_chars > 0 else 0.0,
        'bigram_diversity': unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0,
    }


def filter_text_by_layout(text: str, 
                         layout_mapping: Dict[str, str],
                         replacement_char: str = ' ') -> str:
    """
    Filter text to only include characters that exist in the layout mapping.
    
    Args:
        text: Input text to filter
        layout_mapping: Dictionary mapping characters to positions
        replacement_char: Character to use for unmapped characters
        
    Returns:
        Filtered text with only mapped characters
    """
    if not text or not layout_mapping:
        return ""
    
    # Get set of available characters (case-insensitive)
    available_chars = set(char.lower() for char in layout_mapping.keys())
    
    filtered_chars = []
    for char in text.lower():
        if char in available_chars:
            filtered_chars.append(char)
        elif char.isspace():
            filtered_chars.append(' ')  # Preserve spaces
        else:
            if replacement_char and (not filtered_chars or filtered_chars[-1] != replacement_char):
                filtered_chars.append(replacement_char)
    
    # Join and normalize whitespace
    result = ''.join(filtered_chars)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def calculate_text_coverage(text: str, 
                          layout_mapping: Dict[str, str]) -> Dict[str, float]:
    """
    Calculate how well a layout covers the characters in a text.
    
    Args:
        text: Input text to analyze
        layout_mapping: Dictionary mapping characters to positions
        
    Returns:
        Dictionary with coverage statistics
    """
    if not text:
        return {
            'character_coverage': 0.0,
            'frequency_weighted_coverage': 0.0,
            'unmapped_chars': [],
            'total_chars': 0,
            'mapped_chars': 0,
        }
    
    # Clean text and get character frequencies
    cleaned_text = clean_text_for_analysis(text)
    chars_only = cleaned_text.replace(' ', '')
    
    if not chars_only:
        return {
            'character_coverage': 0.0,
            'frequency_weighted_coverage': 0.0,
            'unmapped_chars': [],
            'total_chars': 0,
            'mapped_chars': 0,
        }
    
    # Get character frequencies
    char_frequencies = get_character_frequencies(chars_only, normalize=True)
    
    # Available characters in layout (case-insensitive)
    available_chars = set(char.lower() for char in layout_mapping.keys())
    
    # Calculate coverage
    total_unique_chars = len(char_frequencies)
    mapped_chars = sum(1 for char in char_frequencies.keys() if char in available_chars)
    unmapped_chars = [char for char in char_frequencies.keys() if char not in available_chars]
    
    # Character coverage (unique characters)
    character_coverage = mapped_chars / total_unique_chars if total_unique_chars > 0 else 0.0
    
    # Frequency-weighted coverage
    mapped_frequency = sum(freq for char, freq in char_frequencies.items() if char in available_chars)
    frequency_weighted_coverage = mapped_frequency
    
    return {
        'character_coverage': character_coverage,
        'frequency_weighted_coverage': frequency_weighted_coverage,
        'unmapped_chars': sorted(unmapped_chars),
        'total_chars': len(chars_only),
        'total_unique_chars': total_unique_chars,
        'mapped_unique_chars': mapped_chars,
        'unmapped_unique_chars': len(unmapped_chars),
    }


def generate_text_summary(text: str, max_length: int = 100) -> str:
    """
    Generate a brief summary of text content.
    
    Args:
        text: Input text to summarize
        max_length: Maximum length of summary
        
    Returns:
        Brief text summary
    """
    if not text:
        return "Empty text"
    
    # Clean text
    cleaned = clean_text_for_analysis(text)
    
    if len(cleaned) <= max_length:
        return cleaned
    
    # Truncate at word boundary
    truncated = cleaned[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can truncate at a reasonable word boundary
        truncated = truncated[:last_space]
    
    return truncated + "..."


def validate_text_input(text: str, 
                       min_length: int = 2,
                       min_unique_chars: int = 1) -> List[str]:
    """
    Validate text input for layout scoring.
    
    Args:
        text: Input text to validate
        min_length: Minimum text length required
        min_unique_chars: Minimum number of unique characters required
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if not text:
        issues.append("Text is empty")
        return issues
    
    # Clean text
    cleaned = clean_text_for_analysis(text)
    chars_only = cleaned.replace(' ', '')
    
    if len(chars_only) < min_length:
        issues.append(f"Text too short: {len(chars_only)} characters (minimum {min_length})")
    
    unique_chars = len(set(chars_only))
    if unique_chars < min_unique_chars:
        issues.append(f"Too few unique characters: {unique_chars} (minimum {min_unique_chars})")
    
    # Check for reasonable character distribution
    if chars_only:
        char_frequencies = get_character_frequencies(chars_only, normalize=True)
        max_frequency = max(char_frequencies.values())
        
        if max_frequency > 0.5:  # Single character dominates
            dominant_char = max(char_frequencies.keys(), key=lambda x: char_frequencies[x])
            issues.append(f"Text dominated by single character '{dominant_char}' ({max_frequency:.1%})")
    
    return issues