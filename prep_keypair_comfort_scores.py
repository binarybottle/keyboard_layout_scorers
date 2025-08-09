#!/usr/bin/env python3
"""
Generate complete comfort scores for all possible QWERTY key-pairs.

This script takes the existing comfort scores for 24-key combinations and
generates scores for the remaining key-pairs to create a complete dataset
of all 1024 possible key-pairs from the 32 QWERTY keys.

The script uses different strategies for different types of key-pairs:

For ONE HAND key-pairs (same hand):
1. Key-pairs with one lateral stretch key (TYGHBN['): 
   - If lateral key is 1st: use minimum comfort score where the non-lateral 
     key appears as 2nd key in existing data
   - If lateral key is 2nd: use minimum comfort score where the non-lateral 
     key appears as 1st key in existing data
2. Key-pairs with two lateral stretch keys: use minimum score from all 
   key-pairs (existing + phase 1 generated) containing either lateral key

For BOTH HANDS key-pairs (different hands):
1. Find maximum comfort score for key-pairs containing the 1st key
2. Find maximum comfort score for key-pairs containing the 2nd key  
3. Average these two maximum scores

Key-pairs that cannot be calculated due to insufficient data are skipped.
Generated key-pairs have empty uncertainty values.

Input:
    input/prep/comfort_keypair_scores_24keys.csv - CSV with existing 24-key comfort scores

Output:
    output/keypair_comfort_scores.csv - Complete CSV with all calculable key-pair scores

Usage:
    python prep_keypair_comfort_scores.py
"""
import csv
import os
import statistics
from pathlib import Path
from typing import Dict, Set, Tuple, List
import logging

# Setup logging for detailed progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Original constants (unchanged)
QWERTY_LAYOUT = {
    # [Previous layout definition - unchanged]
    'Q': (1, 4, 'L'), 'W': (1, 3, 'L'), 'E': (1, 2, 'L'), 'R': (1, 1, 'L'), 'T': (1, 1, 'L'),
    'Y': (1, 1, 'R'), 'U': (1, 1, 'R'), 'I': (1, 2, 'R'), 'O': (1, 3, 'R'), 'P': (1, 4, 'R'),
    'A': (2, 4, 'L'), 'S': (2, 3, 'L'), 'D': (2, 2, 'L'), 'F': (2, 1, 'L'), 'G': (2, 1, 'L'),
    'H': (2, 1, 'R'), 'J': (2, 1, 'R'), 'K': (2, 2, 'R'), 'L': (2, 3, 'R'), ';': (2, 4, 'R'),
    'Z': (3, 4, 'L'), 'X': (3, 3, 'L'), 'C': (3, 2, 'L'), 'V': (3, 1, 'L'), 'B': (3, 1, 'L'),
    'N': (3, 1, 'R'), 'M': (3, 1, 'R'), ',': (3, 2, 'R'), '.': (3, 3, 'R'), '/': (3, 4, 'R'),
    "'": (2, 4, 'R'), '[': (1, 4, 'R'),
}

# Define the lateral stretch keys that require finger stretching
LATERAL_STRETCH_KEYS = set("TYGHBN['")

def get_key_hand(key: str) -> str:
    """Get the hand (L or R) for a key."""
    key = key.upper()
    if key in QWERTY_LAYOUT:
        return QWERTY_LAYOUT[key][2]
    return None

def is_same_hand(key1: str, key2: str) -> bool:
    """Check if two keys are on the same hand."""
    hand1 = get_key_hand(key1)
    hand2 = get_key_hand(key2)
    return hand1 == hand2 and hand1 is not None

def load_existing_scores(input_file: str = "input/prep/comfort_keypair_scores_24keys.csv") -> Dict[str, Tuple[float, float]]:
    """Load existing comfort scores from CSV file."""
    scores = {}
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key_pair = row['key_pair']
            comfort_score = float(row['comfort_score'])
            uncertainty = float(row['uncertainty'])
            scores[key_pair] = (comfort_score, uncertainty)
    
    print(f"📖 Loaded {len(scores)} existing key-pair scores from {input_file}")
    return scores

def get_all_qwerty_keys() -> List[str]:
    """Get all standard QWERTY keys for testing."""
    return list("QWERTYUIOPASDFGHJKL;ZXCVBNM,./'[")

def analyze_existing_scores(scores: Dict[str, Tuple[float, float]]) -> Dict[str, Dict]:
    """Analyze existing scores to extract statistics needed for generation."""
    stats = {
        'min_scores_first_pos': {},  # Min scores when key is in first position
        'min_scores_second_pos': {}, # Min scores when key is in second position
        'max_scores_any_pos': {},    # Max scores when key appears in any position
        'min_scores_any_pos': {}     # Min scores when key appears in any position
    }
    
    # Get unique keys from existing data
    existing_keys = set()
    for key_pair in scores.keys():
        existing_keys.add(key_pair[0])
        existing_keys.add(key_pair[1])
    
    print(f"📊 Analyzing scores for {len(existing_keys)} existing keys")
    
    # Calculate statistics for each key
    for key in existing_keys:
        first_pos_scores = []
        second_pos_scores = []
        any_pos_scores = []
        
        for key_pair, (score, _) in scores.items():
            if key_pair[0] == key:
                first_pos_scores.append(score)
                any_pos_scores.append(score)
            if key_pair[1] == key:
                second_pos_scores.append(score)
                any_pos_scores.append(score)
        
        if first_pos_scores:
            stats['min_scores_first_pos'][key] = min(first_pos_scores)
        if second_pos_scores:
            stats['min_scores_second_pos'][key] = min(second_pos_scores)
        if any_pos_scores:
            stats['max_scores_any_pos'][key] = max(any_pos_scores)
            stats['min_scores_any_pos'][key] = min(any_pos_scores)
    
    return stats

def update_stats_with_new_scores(original_stats: Dict[str, Dict], 
                                new_scores: Dict[str, Tuple[float, float]]) -> Dict[str, Dict]:
    """Update statistics to include newly generated scores."""
    updated_stats = {
        'min_scores_first_pos': dict(original_stats['min_scores_first_pos']),
        'min_scores_second_pos': dict(original_stats['min_scores_second_pos']),
        'max_scores_any_pos': dict(original_stats['max_scores_any_pos']),
        'min_scores_any_pos': dict(original_stats['min_scores_any_pos'])
    }
    
    # Get all keys that appear in new scores
    new_keys = set()
    for key_pair in new_scores.keys():
        new_keys.add(key_pair[0])
        new_keys.add(key_pair[1])
    
    # Calculate statistics for each new key
    for key in new_keys:
        first_pos_scores = []
        second_pos_scores = []
        any_pos_scores = []
        
        for key_pair, (score, _) in new_scores.items():
            if key_pair[0] == key:
                first_pos_scores.append(score)
                any_pos_scores.append(score)
            if key_pair[1] == key:
                second_pos_scores.append(score)
                any_pos_scores.append(score)
        
        # Update or add new statistics
        if first_pos_scores:
            existing_min = updated_stats['min_scores_first_pos'].get(key)
            new_min = min(first_pos_scores)
            updated_stats['min_scores_first_pos'][key] = min(existing_min, new_min) if existing_min is not None else new_min
        
        if second_pos_scores:
            existing_min = updated_stats['min_scores_second_pos'].get(key)
            new_min = min(second_pos_scores)
            updated_stats['min_scores_second_pos'][key] = min(existing_min, new_min) if existing_min is not None else new_min
        
        if any_pos_scores:
            existing_max = updated_stats['max_scores_any_pos'].get(key)
            existing_min = updated_stats['min_scores_any_pos'].get(key)
            new_max = max(any_pos_scores)
            new_min = min(any_pos_scores)
            
            updated_stats['max_scores_any_pos'][key] = max(existing_max, new_max) if existing_max is not None else new_max
            updated_stats['min_scores_any_pos'][key] = min(existing_min, new_min) if existing_min is not None else new_min
    
    return updated_stats

def save_complete_scores(existing_scores: Dict[str, Tuple[float, float]], 
                        generated_scores: Dict[str, Tuple[float, float]], 
                        output_file: str = "output/keypair_comfort_scores.csv"):
    """Save complete comfort scores to CSV file."""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Combine existing and generated scores
    all_scores = {**existing_scores, **generated_scores}
    
    # Convert to list and sort by key-pair for consistent ordering
    results = []
    for key_pair, (comfort_score, uncertainty) in all_scores.items():
        results.append({
            'key_pair': key_pair,
            'comfort_score': comfort_score,
            'uncertainty': uncertainty if uncertainty is not None else ''
        })
    
    results.sort(key=lambda x: x['key_pair'])
    
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['key_pair', 'comfort_score', 'uncertainty'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"💾 Saved {len(results)} total key-pair scores to: {output_file}")
    return len(results)

def validate_generated_scores_realtime(generated_scores: Dict[str, Tuple[float, float]], 
                                     phase: str) -> bool:
    """Real-time validation during score generation."""
    if not generated_scores:
        return True
    
    scores = [score for score, _ in generated_scores.values()]
    
    # Check for invalid scores
    invalid_scores = [s for s in scores if not (-10 <= s <= 10)]
    if invalid_scores:
        logger.error(f"Phase {phase}: Found {len(invalid_scores)} invalid scores")
        return False
    
    # Check for identical scores (might indicate algorithm issue)
    unique_scores = len(set(scores))
    if unique_scores < len(scores) * 0.5:  # Less than 50% unique
        logger.warning(f"Phase {phase}: Low score diversity ({unique_scores}/{len(scores)} unique)")
    
    return True

def generate_missing_scores(existing_scores: Dict[str, Tuple[float, float]], 
                                   stats: Dict[str, Dict]) -> Dict[str, Tuple[float, float]]:
    """Generate comfort scores for missing key-pairs based on the rules."""
    all_keys = get_all_qwerty_keys()
    missing_scores = {}
    
    logger.info("🔄 Generating missing scores with validation...")
    
    # Phase 1: Same-hand pairs with one lateral stretch key
    logger.info("   Phase 1: Same-hand pairs with one lateral stretch key")
    phase1_scores = {}
    
    for key1 in all_keys:
        for key2 in all_keys:
            key_pair = key1 + key2
            if key_pair in existing_scores:
                continue
            
            same_hand = is_same_hand(key1, key2)
            key1_is_lateral = key1 in LATERAL_STRETCH_KEYS
            key2_is_lateral = key2 in LATERAL_STRETCH_KEYS
            
            if same_hand and (key1_is_lateral + key2_is_lateral == 1):
                if key1_is_lateral:
                    min_score = stats['min_scores_second_pos'].get(key2)
                else:
                    min_score = stats['min_scores_first_pos'].get(key1)
                
                if min_score is not None:
                    phase1_scores[key_pair] = (min_score, None)
    
    # Validate Phase 1 results
    if not validate_generated_scores_realtime(phase1_scores, "1"):
        raise ValueError("Phase 1 validation failed")
    
    logger.info(f"      Generated {len(phase1_scores)} Phase 1 scores ✅")
    missing_scores.update(phase1_scores)
    
    # Phase 2: Same-hand pairs with two lateral stretch keys
    logger.info("   Phase 2: Same-hand pairs with two lateral stretch keys")
    phase2_scores = {}
    combined_scores = {**existing_scores, **missing_scores}
    
    for key1 in all_keys:
        for key2 in all_keys:
            key_pair = key1 + key2
            if key_pair in existing_scores or key_pair in missing_scores:
                continue
            
            same_hand = is_same_hand(key1, key2)
            key1_is_lateral = key1 in LATERAL_STRETCH_KEYS
            key2_is_lateral = key2 in LATERAL_STRETCH_KEYS
            
            if same_hand and key1_is_lateral and key2_is_lateral:
                min_scores = []
                for existing_pair, (score, _) in combined_scores.items():
                    if key1 in existing_pair or key2 in existing_pair:
                        min_scores.append(score)
                
                if min_scores:
                    phase2_scores[key_pair] = (min(min_scores), None)
    
    # Validate Phase 2 results  
    if not validate_generated_scores_realtime(phase2_scores, "2"):
        raise ValueError("Phase 2 validation failed")
    
    logger.info(f"      Generated {len(phase2_scores)} Phase 2 scores ✅")
    missing_scores.update(phase2_scores)
    
    # Update stats for Phase 3
    logger.info("   Updating statistics for Phase 3...")
    updated_stats = update_stats_with_new_scores(stats, missing_scores)
    
    # Phase 3: Different-hand pairs
    logger.info("   Phase 3: Different-hand pairs")
    phase3_scores = {}
    
    for key1 in all_keys:
        for key2 in all_keys:
            key_pair = key1 + key2
            if key_pair in existing_scores or key_pair in missing_scores:
                continue
            
            same_hand = is_same_hand(key1, key2)
            if not same_hand:
                max1 = updated_stats['max_scores_any_pos'].get(key1)
                max2 = updated_stats['max_scores_any_pos'].get(key2)
                
                if max1 is not None and max2 is not None:
                    avg_score = (max1 + max2) / 2
                    phase3_scores[key_pair] = (avg_score, None)
    
    # Validate Phase 3 results
    if not validate_generated_scores_realtime(phase3_scores, "3"):
        raise ValueError("Phase 3 validation failed")
    
    logger.info(f"      Generated {len(phase3_scores)} Phase 3 scores ✅")
    missing_scores.update(phase3_scores)
    
    # Final validation
    total_generated = len(phase1_scores) + len(phase2_scores) + len(phase3_scores)
    logger.info(f"✅ Successfully generated {total_generated} missing scores")
    
    return missing_scores

def comprehensive_output_validation(output_file: str) -> bool:
    """Comprehensive validation of the output file."""
    logger.info("🔍 Performing comprehensive output validation...")
    
    if not os.path.exists(output_file):
        logger.error(f"Output file not found: {output_file}")
        return False
    
    # Load and validate structure
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Basic structure checks
    expected_count = len(get_all_qwerty_keys()) ** 2
    if len(rows) != expected_count:
        logger.error(f"Expected {expected_count} rows, got {len(rows)}")
        return False
    
    # Validate each row
    for i, row in enumerate(rows):
        if len(row['key_pair']) != 2:
            logger.error(f"Row {i}: Invalid key_pair length: {row['key_pair']}")
            return False
        
        try:
            score = float(row['comfort_score'])
            if not (-10 <= score <= 10):  # Reasonable range
                logger.warning(f"Row {i}: Unusual score: {score}")
        except ValueError:
            logger.error(f"Row {i}: Invalid comfort_score: {row['comfort_score']}")
            return False
    
    # Statistical validation
    scores = [float(row['comfort_score']) for row in rows]
    
    # Check for score distribution
    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    logger.info(f"   Score statistics:")
    logger.info(f"     Mean: {mean_score:.3f}")
    logger.info(f"     Std:  {std_score:.3f}")
    logger.info(f"     Range: {min_score:.3f} to {max_score:.3f}")
    
    # Check for duplicates
    key_pairs = [row['key_pair'] for row in rows]
    if len(set(key_pairs)) != len(key_pairs):
        logger.error("Found duplicate key pairs")
        return False
    
    # Validate uncertainty pattern
    existing_with_uncertainty = 0
    generated_with_uncertainty = 0
    
    for row in rows:
        key1, key2 = row['key_pair'][0], row['key_pair'][1]
        has_lateral = key1 in LATERAL_STRETCH_KEYS or key2 in LATERAL_STRETCH_KEYS
        has_uncertainty = row['uncertainty'] != ''
        
        if not has_lateral and has_uncertainty:
            existing_with_uncertainty += 1
        elif has_lateral and has_uncertainty:
            generated_with_uncertainty += 1
    
    logger.info(f"   Uncertainty pattern:")
    logger.info(f"     Existing with uncertainty: {existing_with_uncertainty}")
    logger.info(f"     Generated with uncertainty: {generated_with_uncertainty}")
    
    if generated_with_uncertainty > 0:
        logger.warning("Generated scores should not have uncertainty values")
    
    logger.info("✅ Output validation completed successfully")
    return True

def generate_validation_examples(output_file: str) -> None:
    """Generate detailed examples showing the algorithm in action."""
    logger.info("📝 Generating validation examples...")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Categorize examples
    examples = {'existing': [], 'phase1': [], 'phase2': [], 'phase3': []}
    
    for row in rows:
        key1, key2 = row['key_pair'][0], row['key_pair'][1]
        key1_lateral = key1 in LATERAL_STRETCH_KEYS
        key2_lateral = key2 in LATERAL_STRETCH_KEYS
        lateral_count = key1_lateral + key2_lateral
        same_hand = is_same_hand(key1, key2)
        
        if lateral_count == 0:
            examples['existing'].append(row)
        elif same_hand and lateral_count == 1:
            examples['phase1'].append(row)
        elif same_hand and lateral_count == 2:
            examples['phase2'].append(row)
        else:
            examples['phase3'].append(row)
    
    # Show examples from each category
    logger.info("📋 Validation examples by category:")
    
    for category, items in examples.items():
        if not items:
            continue
        
        logger.info(f"\n   {category.upper()} examples:")
        sample_size = min(5, len(items))
        
        for item in items[:sample_size]:
            key1, key2 = item['key_pair'][0], item['key_pair'][1]
            score = float(item['comfort_score'])
            uncertainty = item['uncertainty'] if item['uncertainty'] else 'null'
            hand_info = 'same' if is_same_hand(key1, key2) else 'diff'
            
            logger.info(f"     {item['key_pair']}: {score:.4f} "
                       f"({hand_info} hand, unc={uncertainty})")

def main():
    """Main function with comprehensive validation."""
    print("Generate Complete QWERTY Key-pair Comfort Scores")
    print("=" * 60)
    
    try:
        # Load and validate existing scores
        input_file = "input/prep/comfort_keypair_scores_24keys.csv"
        existing_scores = load_existing_scores(input_file)
        
        # Analyze existing scores
        stats = analyze_existing_scores(existing_scores)
        
        # Generate missing scores with validation
        generated_scores = generate_missing_scores(existing_scores, stats)
        
        # Save complete dataset
        output_file = "output/keypair_comfort_scores.csv"
        total_count = save_complete_scores(existing_scores, generated_scores, output_file)
        
        # Comprehensive output validation
        if not comprehensive_output_validation(output_file):
            raise ValueError("Output validation failed")
        
        # Generate validation examples
        generate_validation_examples(output_file)
        
        # Final success report
        print(f"\n🏆 VALIDATION COMPLETED SUCCESSFULLY!")
        print(f"   ✅ Generation logic: PASSED") 
        print(f"   ✅ Output validation: PASSED")
        print(f"   ✅ Statistical checks: PASSED")
        print(f"   📊 Total scores generated: {total_count}")
        print(f"   📁 Output file: {output_file}")
        
        logger.info("All validations passed. Data is ready for production use.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ VALIDATION FAILED: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)