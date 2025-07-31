# Keyboard Layout Scorer
Framework for evaluating keyboard layouts using multiple scoring methods 

**Repository**: https://github.com/binarybottle/keyboard_layout_scorers.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

This framework unifies several keyboard layout scoring approaches under a common architecture. Their scores are not directly comparable with each other since they measure fundamentally different aspects of layout quality, but it is useful to see how the same scoring method evaluates different layouts:

- [Distance Scorer](distance_scorer.py): Physical finger travel distance analysis
- [Dvorak-9 Scorer](dvorak9_scorer.py): Four scoring approaches (pure, and frequency-, speed-, and comfort-weighted)
- [Engram Scorer](engram_scorer.py): Comfort + frequency scoring (32-key and 24-key home block only)
  - GitHub: https://github.com/binarybottle/optimize_layouts
  - OSF: https://osf.io/6dt75/  (DOI: 10.17605/OSF.IO/6DT75)

For reference:
- QWERTY:  "qwertyuiopasdfghjkl;zxcvbnm,./"  
- Dvorak:  "',.pyfgcrlaoeuidhtns;qjkxbmwvz"
- Colemak: "qwfpgjluy;arstdhneiozxcvbkm,./"

## Installation
  ```bash
  # Install dependencies
  pip install pyyaml pandas numpy

  # Or with poetry
  poetry install
  ```

## Quick Start

### Single-scorer, single-layout mode
  ```bash
  # Distance scorer (requires text input)
  python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"

  # Dvorak9 scorer (shows all 4 approaches automatically)
  python layout_scorer.py --scorer dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

  # Engram scorer (shows both 32-key and 24-key results automatically)
  python layout_scorer.py --scorer engram --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

  # With cross-hand filtering (available for all scorers)
  python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello" --ignore-cross-hand

  # Individual scorers
  python distance_scorer.py --letters "abc" --positions "QWE" --text "hello world"
  python dvorak_scorer.py --letters "abc" --positions "QWE"
  python engram_scorer.py --letters "abc" --positions "QWE"
  ```

### Multiple-scorer, single-layout mode
  ```bash
  # Run all scorers
  python layout_scorer.py --scorers all --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"

  # Run specific subset
  python layout_scorer.py --scorers engram,dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
  ```

### Multiple-scorer, multiple-layout comparison mode
  ```bash
  # Compare layouts (shows both full layout and common-keys-only results)
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;" --text "hello world"

  # Save detailed comparison to CSV
  python layout_scorer.py --compare qwerty:"qwertyuiopasdfghjkl;zxcvbnm,./" dvorak:"',.pyfgcrlaoeuidhtns;qjkxbmwvz" --csv results.csv --text "hello"
  ```

## Output Formats

### Detailed output (default)
Shows comprehensive results with breakdowns, validation info, and scoring details:
  ```bash
  python layout_scorer.py --scorer dvorak9 --letters "abc" --positions "ABC" --detailed
  ```

### CSV output
  ```bash
  # Single layout to CSV
  python layout_scorer.py --scorer engram --letters "abc" --positions "ABC" --csv results.csv

  # Comparison to CSV (includes both full and common-keys-only results)
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv compare.csv --text "hello"
  ```

### Score-only output
Outputs just the numeric scores in space-separated format:
  ```bash
  python layout_scorer.py --scorer distance --letters "abc" --positions "ABC" --text "hello" --score-only
  ```

## Advanced Features

### Cross-hand filtering
Filters out bigrams that cross hands (left-to-right or right-to-left transitions). Available for all scorers:
  ```bash
  # Apply to any scorer
  python layout_scorer.py --scorer [distance|dvorak9|engram] --letters "..." --positions "..." --ignore-cross-hand
  ```

### Common-keys-only Comparison
When comparing multiple layouts, automatically shows both:
  1. **Full layout results**: All keys in each layout
  2. **Common-keys-only results**: Only keys present in all compared layouts

  ```bash
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;" --text "hello"
  # Outputs: qwerty, dvorak, colemak, qwerty_common, dvorak_common, colemak_common
  ```

## Configuration
The framework uses `config.yaml` for centralized configuration:

  ```yaml
  common:
    default_output_format: "detailed"
    letter_filtering: true
    data_directories:
      base: "input/"
      engram: "input/engram/"
      dvorak9: "input/dvorak9/"

  distance_scorer:
    description: "Physical finger travel distance analysis with cross-hand filtering support"
    scoring_options:
      ignore_cross_hand: false

  dvorak9_scorer:
    description: "Dvorak's 9 theoretical typing principles with four scoring approaches"
    data_files:
      frequencies: "normalized_letter_pair_frequencies_en.csv"
      speed_weights: "speed_weights.csv"      # automatically loaded if available
      comfort_weights: "comfort_weights.csv"  # automatically loaded if available
    scoring_options:
      ignore_cross_hand: false

  engram_scorer:
    description: "Letter frequency and key comfort combination scoring with 32-key and 24-key modes"
    scoring_options:
      ignore_cross_hand: false
      home_block_keys: "qwerasdfzxcvuiopjkl;m,./"
  ```

## Data Files

### Distance Scorer
  - **No external files required** (uses built-in position mapping)

### Dvorak-9 Scorer
**Required:**
  - `input/dvorak9/normalized_letter_pair_frequencies_en.csv` - English bigram frequencies
  - `input/dvorak9/key_pair_scores.csv` - Precomputed Dvorak-9 scores

**Optional (automatically loaded if available):**
  - `input/dvorak9/speed_weights.csv` - Speed-based empirical weights
  - `input/dvorak9/comfort_weights.csv` - Comfort-based empirical weights

### Engram Scorer
**Required:**
  - `input/engram/normalized_letter_frequencies_en.csv` - Letter frequencies
  - `input/engram/normalized_letter_pair_frequencies_en.csv` - Bigram frequencies  
  - `input/engram/normalized_key_comfort_scores_24keys.csv` - Key comfort scores
  - `input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv` - Key-pair comfort

### File Formats
All data files use CSV format with headers:

  **Frequency files:**
  ```csv
  letter,frequency
  e,12.70
  t,9.06
  ```

  **Bigram frequency files:**
  ```csv
  letter_pair,frequency
  th,3.56
  he,3.07
  ```

  **Score files:**
  ```csv
  key,comfort_score
  D,7.2
  F,8.1
  ```

## Architecture
  ```
  keyboard_layout_scorers/
  ├── config.yaml                    # Central configuration
  ├── README.md                      # This file
  ├── layout_scorer.py               # Unified manager (main tool)
  ├── 
  ├── # Individual scorers
  ├── distance_scorer.py             # Physical distance scoring
  ├── dvorak9_scorer.py              # Dvorak-9 theoretical scoring  
  ├── engram_scorer.py               # Frequency-comfort scoring
  ├──
  ├── # Framework utilities  
  ├── framework/
  │   ├── __init__.py                # Package initialization
  │   ├── config_loader.py           # Configuration management
  │   ├── base_scorer.py             # Abstract base classes  
  │   ├── layout_utils.py            # Layout utilities
  │   ├── data_utils.py              # Data loading utilities
  │   ├── text_utils.py              # Text processing utilities
  │   ├── output_utils.py            # Output formatting utilities
  │   ├── cli_utils.py               # CLI utilities
  │   ├── scorer_factory.py          # Scorer factory class
  │   └── unified_scorer.py          # Unified scorer manager
  ├──
  ├── # Data directories
  ├── input/
  │   ├── engram/                    # Engram scorer data files
  │   ├── dvorak9/                   # Dvorak-9 scorer data files  
  │   └── distance/                  # Distance scorer data files (if any)
  └──
  ```

### Class Hierarchy
  ```python
  BaseLayoutScorer (ABC)
  ├── DistanceScorer
  ├── Dvorak9Scorer  
  └── EngramScorer

  ScoreResult (dataclass)
  ├── primary_score: float
  ├── components: Dict[str, float]
  ├── metadata: Dict[str, Any]
  ├── validation_info: Dict[str, Any]
  ├── detailed_breakdown: Dict[str, Any]
  └── extract_all_metrics() -> Dict[str, float]

  ScorerFactory
  ├── create_scorer()
  └── get_available_scorers()

  UnifiedLayoutScorer
  ├── score_layout()
  └── compare_layouts()
  ```

## Output Examples

### Dvorak-9 Scorer Output
  ```
  Dvorak9 Scorer Results
  ======================================================================
  Primary score:     0.750000

  Component scores:
    Pure Dvorak Score              :   0.720000
    Frequency Weighted Score       :   0.750000
    Speed Weighted Score           :   0.680000
    Comfort Weighted Score         :   0.710000
    Hands                          :   0.850000
    Fingers                        :   0.920000
    [... other criteria ...]

  Execution time: 0.045 seconds
  ```

### Engram Scorer Output  
  ```
  Engram Scorer Results
  ======================================================================
  Primary score:     2.450000

  Component scores:
    Total Score 32key              :   2.450000
    Total Score 24key              :   1.890000
    Item Component 32key           :   0.720000
    Item Pair Component 32key      :   0.850000
    Item Component 24key           :   0.680000
    Item Pair Component 24key      :   0.790000

  Execution time: 0.032 seconds
  ```

### Layout Comparison Output
  ```
  === FULL LAYOUT COMPARISON ===
  Layout: qwerty vs dvorak vs colemak

  === COMMON KEYS ONLY COMPARISON ===
  Common keys (8): aeinorst

  Layout Scoring Summary
  ======================================================================
  Rank   Layout               Primary score    Scorer
  #1     dvorak               0.750000         Dvorak9
  #2     colemak              0.720000         Dvorak9  
  #3     qwerty               0.680000         Dvorak9
  #1     dvorak_common        0.720000         Dvorak9
  #2     colemak_common       0.700000         Dvorak9
  #3     qwerty_common        0.650000         Dvorak9
  ```

## Testing and Validation
The framework includes a comprehensive validation script:

```bash
python validate_layout_scorer.py

# Test specific script location
python validate_layout_scorer.py --script /path/to/layout_scorer.py

# Save detailed report
python validate_layout_scorer.py --report validation_report.json

# Run quietly (less verbose output)
python validate_layout_scorer.py --quiet
```