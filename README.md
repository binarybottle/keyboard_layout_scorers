# Keyboard Layout Scorer
Framework for evaluating keyboard layouts using multiple scoring methods 

**Repository**: https://github.com/binarybottle/keyboard_layout_scorers.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

This framework unifies several keyboard layout scoring approaches under a common architecture. Their scores are not directly comparable with each other since they measure fundamentally different aspects of layout quality, but it is useful to see how the same scoring method evaluates different layouts:
  - [Distance Scorer](distance_scorer.py): Physical finger travel distance analysis
  - [Dvorak-9 Scorer](dvorak9_scorer.py):  Theoretical typing principles with empirical validation  
  - [Engram Scorer](engram_scorer.py):     Letter frequency and key comfort combination scoring
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

## Usage

### Single-scorer mode
  ```bash
  # Uniform command: 
  python layout_scorer.py --scorer distance --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"
  python layout_scorer.py --scorer dvorak9  --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
  python layout_scorer.py --scorer engram   --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

  # Individual scorers:
  python distance_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"
  python dvorak9_scorer.py  --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"  
  python engram_scorer.py   --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

  # Dvorak-9: weighted scoring
  python layout_scorer.py --scorer dvorak9 --letters "abc" --positions "QWE" \
    --weights "input/dvorak9/speed_weights.csv"
  python layout_scorer.py --scorer dvorak9 --letters "abc" --positions "QWE" \
    --weights "input/dvorak9/comfort_weights.csv"
  python dvorak9_scorer.py --letters "abc" --qwerty_keys "QWE" \
    --weights "input/dvorak9/speed_weights.csv"

  # Engram: ignore cross-hand bigrams
  python layout_scorer.py --scorer engram --letters "abc" --positions "QWE" --ignore-cross-hand
  python engram_scorer.py --letters "abc" --positions "QWE" --ignore-cross-hand
  ```

### Multiple-scorer mode
  ```bash
  # Run all scorers
  python layout_scorer.py --scorers all --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello world"

  # Run specific subset
  python layout_scorer.py --scorers engram,dvorak9 --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "hello"
  ```

### Layout comparison mode
  ```bash
  # Compare layouts (layout string maps to QWERTY positions sequentially)
  # `dvorak:"',.pyfgcrl"` → '→Q, ,→W, .→E, p→R, y→T, f→Y, g→U, c→I, r→O, l→P
  python layout_scorer.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" colemak:"qwfpgjluy;" --text "hello world"
  ```

### Output formats
  ```bash
  # Detailed output (default)
  python layout_scorer.py --scorer engram --letters "abc" --positions "ABC" --detailed

  # CSV output
  python layout_scorer.py --scorer engram --letters "abc" --positions "ABC" --csv single_result.csv

  # Save detailed comparison to CSV file
  python layout_scorer.py --compare qwerty:"qwertyuiopasdfghjkl;zxcvbnm,./" dvorak:"',.pyfgcrlaoeuidhtns;qjkxbmwvz" colemak:"qwfpgjluy;arstdhneiozxcvbkm,./" --csv compare3layouts.csv --text "hello"

  # Score only
  python layout_scorer.py --scorer distance --letters "abc" --positions "ABC" --text "hello" --score-only
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
    description: "Physical finger travel distance analysis"
    method: "Euclidean distance between key positions for finger transitions"
    data_files:
      position_map: null  # Built-in data
    scoring_options:
      distance_metric: "euclidean"
    output:
      primary_score_name: "normalized_score"
      components: ["total_distance", "average_distance", "coverage"]
  ...
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

### Data file formats
Required files by scorer:

  **Distance scorer**: No external files (uses built-in position data)

  **Dvorak-9 scorer**:
  - `input/dvorak9/normalized_letter_pair_frequencies_en.csv` - English bigram frequencies
  - `input/dvorak9/key_pair_scores.csv` - Precomputed Dvorak-9 scores  
  - `input/dvorak9/speed_weights.csv` - Speed-based empirical weights (optional)
  - `input/dvorak9/comfort_weights.csv` - Comfort-based empirical weights (optional)

  **Engram scorer**:
  - `input/engram/normalized_letter_frequencies_en.csv` - Letter frequencies
  - `input/engram/normalized_letter_pair_frequencies_en.csv` - Bigram frequencies
  - `input/engram/normalized_key_comfort_scores_24keys.csv` - Key comfort scores
  - `input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv` - Key-pair comfort

  All data files use CSV format with headers:

  **Frequency files**:
  ```csv
  letter,frequency
  e,12.70
  t,9.06
  ```

  **Bigram frequency files**:
  ```csv
  letter_pair,frequency
  th,3.56
  he,3.07
  ```

  **Score files**:
  ```csv
  key,comfort_score
  D,7.2
  F,8.1
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
  └── extract_all_metrics() -> Dict[str, float]  # NEW

  ScorerFactory
  ├── create_scorer()
  └── get_available_scorers()

  UnifiedLayoutScorer
  ├── score_layout()
  └── compare_layouts()
  ```

## Testing and Validation
The framework includes a comprehensive validation script:

  ```bash
  python validate_layout_scorer.py
  ```