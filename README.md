# Unified Keyboard Layout Scoring Framework

A comprehensive, extensible framework for evaluating keyboard layouts using multiple scoring methodologies. This system provides standardized interfaces, consistent output formats, and easy extensibility for adding new scoring methods.

## Overview

This framework unifies several keyboard layout scoring approaches under a common architecture:

- **Distance Scorer**: Physical finger travel distance analysis
- **Dvorak-9 Scorer**: Theoretical typing principles with empirical validation  
- **Engram Scorer**: Letter frequency and key comfort combination scoring

### Key Features

 **Standardized CLI** - Consistent argument names and formats across all scorers  
 **Configuration-driven** - YAML configuration for easy customization  
 **Multiple output formats** - CSV, detailed, and score-only modes  
 **Modular architecture** - Shared utilities and extensible design  
 **Type safety** - Structured result objects and validation  
 **Error handling** - Comprehensive validation and helpful error messages  

## Quick Start

### Installation

```bash
# Install dependencies
pip install pyyaml pandas numpy

# Or with poetry
poetry install
```

### Basic Usage

```bash
# Distance-based scoring
python distance_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --text "the quick brown fox"

# Dvorak-9 theoretical scoring
python dvorak9_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Engram frequency-comfort scoring  
python engram_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
```

### Output Formats

```bash
# Detailed output (default)
python distance_scorer.py --letters "abc" --positions "ABC" --text "hello" --detailed

# CSV output
python distance_scorer.py --letters "abc" --positions "ABC" --text "hello" --csv

# Score only
python distance_scorer.py --letters "abc" --positions "ABC" --text "hello" --score-only
```

## Scoring Methods

### Distance Scorer

**Purpose**: Calculates physical finger travel distance on staggered keyboard layouts

**Method**: Euclidean distance between key positions for consecutive character pairs (bigrams)

**Input**: Layout mapping + text to analyze

**Output**: 
- Primary score: Normalized distance score (higher = better)
- Components: Total distance, average distance, coverage metrics

**Example**:
```bash
python distance_scorer.py \
  --letters "etaoinshrlcu" \
  --positions "FDESGJWXRTYZ" \
  --text "the quick brown fox jumps over the lazy dog"
```

### Dvorak-9 Scorer

**Purpose**: Evaluates layouts using Dvorak's 9 theoretical typing principles

**Method**: Frequency-weighted evaluation of hand alternation, finger usage, etc.

**Scoring approaches**:
- Pure Dvorak: Unweighted average of 9 criteria
- Frequency-weighted: English bigram frequency-weighted
- Empirically-weighted: Real performance data weighted

**The 9 Criteria** (0-1 scale, higher = better):
1. **Hands** - Favor alternating hands over same hand
2. **Fingers** - Avoid same finger repetition  
3. **Skip fingers** - Favor non-adjacent fingers over adjacent
4. **Don't cross home** - Avoid crossing over the home row
5. **Same row** - Favor typing within the same row
6. **Home row** - Favor using the home row
7. **Columns** - Favor fingers staying in designated columns
8. **Strum** - Favor inward rolls over outward rolls
9. **Strong fingers** - Favor stronger fingers over weaker ones

**Example**:
```bash
# Basic scoring
python dvorak9_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# With empirical speed weights
python dvorak9_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" \
  --weights "input/dvorak9/speed_weights.csv"
```

### Engram Scorer

**Purpose**: Combines letter frequencies with key comfort scores

**Method**: Multiplicative combination of item frequencies and position comfort

**Input**: Layout mapping + frequency/comfort data files

**Output**:
- Primary score: Combined frequency-comfort score
- Components: Item component, item-pair component

**Example**:
```bash
# Basic scoring
python engram_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Ignore cross-hand bigrams
python engram_scorer.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --ignore-cross-hand
```

## Configuration System

The framework uses `config.yaml` for centralized configuration:

### Structure

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
  method: "Euclidean distance between key positions"
  data_files:
    position_map: null  # Built-in data
  scoring_options:
    distance_metric: "euclidean"
  output:
    primary_score_name: "normalized_score"
    components: ["total_distance", "average_distance", "coverage"]

# Similar sections for dvorak9_scorer and engram_scorer...
```

### Customization

1. **Data file paths**: Update `data_directories` and `data_files`
2. **Scoring options**: Modify `scoring_options` for each scorer
3. **Output format**: Change default output format and precision
4. **Validation**: Adjust validation thresholds

## Architecture

### Core Components

```
config.yaml              # Central configuration
config_loader.py          # Configuration management
base_scorer.py           # Abstract base classes
layout_utils.py          # Layout manipulation utilities
data_utils.py            # Data loading and normalization
text_utils.py            # Text processing utilities
output_utils.py          # Output formatting
cli_utils.py             # Command-line interface utilities
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
└── detailed_breakdown: Dict[str, Any]
```

### Data Flow

1. **CLI parsing** → Standardized arguments across all scorers
2. **Config loading** → Merge common + scorer-specific settings  
3. **Layout validation** → Ensure mapping correctness
4. **Data loading** → CSV files, frequency data, etc.
5. **Score calculation** → Scorer-specific methodology
6. **Result formatting** → Consistent output across formats

## Adding New Scorers

The framework makes it easy to add new scoring methods:

### 1. Create Scorer Class

```python
from base_scorer import BaseLayoutScorer, ScoreResult

class MyNewScorer(BaseLayoutScorer):
    def load_data_files(self) -> None:
        # Load any required CSV files
        pass
    
    def calculate_scores(self) -> ScoreResult:
        # Implement your scoring logic
        return ScoreResult(
            primary_score=score,
            components={"component1": value1, "component2": value2},
            metadata={"method": "my_method"}
        )
```

### 2. Add Configuration

Add to `config.yaml`:

```yaml
my_new_scorer:
  description: "Description of your scoring method"
  method: "How it works"
  data_files:
    required_file: "path/to/data.csv"
  scoring_options:
    option1: "value1"
  output:
    primary_score_name: "my_score"
    components: ["component1", "component2"]
```

### 3. Create CLI Script

```python
from cli_utils import create_standard_parser, handle_common_errors

@handle_common_errors  
def main():
    cli_parser = create_standard_parser('my_new_scorer')
    args = cli_parser.parse_args()
    
    # Standard framework workflow
    config = load_scorer_config('my_new_scorer', args.config)
    letters, positions, layout_mapping = get_layout_from_args(args)
    
    scorer = MyNewScorer(layout_mapping, config)
    result = scorer.score_layout()
    
    print_results(result, args.output_format)

if __name__ == "__main__":
    sys.exit(main())
```

## Data Files

### Required Files by Scorer

**Distance Scorer**: No external files (uses built-in position data)

**Dvorak-9 Scorer**:
- `input/dvorak9/normalized_letter_pair_frequencies_en.csv` - English bigram frequencies
- `input/dvorak9/key_pair_scores.csv` - Precomputed Dvorak-9 scores  
- `input/dvorak9/speed_weights.csv` - Speed-based empirical weights (optional)
- `input/dvorak9/comfort_weights.csv` - Comfort-based empirical weights (optional)

**Engram Scorer**:
- `input/engram/normalized_letter_frequencies_en.csv` - Letter frequencies
- `input/engram/normalized_letter_pair_frequencies_en.csv` - Bigram frequencies
- `input/engram/normalized_key_comfort_scores_32keys.csv` - Key comfort scores
- `input/engram/normalized_key_pair_comfort_scores_32keys_LvsRpairs.csv` - Key-pair comfort

### File Formats

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

## Migration from Original Scorers

### Original vs Framework

| Aspect | Original | Framework |
|--------|----------|-----------|
| **CLI args** | Inconsistent (`--qwerty-keys` vs `--positions`) | Standardized (`--positions`) |
| **Output** | Custom formatting | Standardized (CSV/detailed/score-only) |
| **Config** | Hardcoded | YAML configuration |
| **Code reuse** | Duplicated utilities | Shared framework |
| **Validation** | Limited | Comprehensive |
| **Error handling** | Basic | Standardized |

### Argument Mapping

```bash
# Original distance scorer
python distance_scorer.py --letters "abc" --qwerty-keys "ABC" --text "hello"

# Framework version  
python distance_scorer.py --letters "abc" --positions "ABC" --text "hello"
```

```bash
# Original dvorak9 scorer
python dvorak9_scorer.py --letters "abc" --qwerty_keys "ABC"

# Framework version
python dvorak9_scorer.py --letters "abc" --positions "ABC"
```

```bash
# Original engram scorer  
python engram_scorer.py --items "abc" --positions "ABC"

# Framework version
python engram_scorer.py --letters "abc" --positions "ABC"
```

## Development

### Project Structure

```
keyboard_layout_scorers/
├── config.yaml                    # Central configuration
├── README.md                      # This file
├── 
├── # Framework core
├── config_loader.py               # Configuration management
├── base_scorer.py                 # Abstract base classes  
├── layout_utils.py                # Layout utilities
├── data_utils.py                  # Data loading utilities
├── text_utils.py                  # Text processing utilities
├── output_utils.py                # Output formatting utilities
├── cli_utils.py                   # CLI utilities
├── 
├── # Scorers (framework versions)
├── distance_scorer.py             # Physical distance scoring
├── dvorak9_scorer.py              # Dvorak-9 theoretical scoring
├── engram_scorer.py               # Frequency-comfort scoring
├──
└── # Data directories
    input/
    ├── engram/                    # Engram scorer data files
    ├── dvorak9/                   # Dvorak-9 scorer data files  
    └── distance/                  # Distance scorer data files (if any)
```

### Testing

```bash
# Test each scorer with sample data
python distance_scorer.py --letters "etaoin" --positions "FDESGJ" --text "hello world"
python dvorak9_scorer.py --letters "etaoin" --positions "FDESGJ"  
python engram_scorer.py --letters "etaoin" --positions "FDESGJ"

# Test different output formats
python distance_scorer.py --letters "etaoin" --positions "FDESGJ" --text "hello" --csv
python distance_scorer.py --letters "etaoin" --positions "FDESGJ" --text "hello" --score-only
```

### Contributing

1. **Add new scorer**: Follow the pattern in "Adding New Scorers"
2. **Extend utilities**: Add functions to appropriate utility modules
3. **Update config**: Add configuration for new features
4. **Test thoroughly**: Ensure compatibility across all scorers

## Troubleshooting

### Common Issues

**"No valid bigrams found"**
- Check that layout mapping characters match text characters
- Ensure case consistency (framework converts to lowercase)
- Verify text is not empty after cleaning

**"Configuration file not found"**
- Ensure `config.yaml` is in the current directory
- Use `--config path/to/config.yaml` to specify custom location

**"Data file not found"**
- Check that required data files exist in specified paths
- Update `data_directories` in config.yaml if files are elsewhere
- Some scorers require specific CSV files (see "Data Files" section)

**"Layout validation warnings"**
- Review character-to-position mapping for duplicates
- Ensure all characters are valid keyboard positions
- Use `--quiet` to suppress non-critical warnings

### Performance Tips

- **Large text files**: Use `--score-only` for faster processing
- **Batch processing**: Consider creating scripts that call scorers programmatically
- **Data caching**: Framework caches loaded data files within single runs

## References

### Original Implementations
- [Distance Scorer Documentation](README_distance.md)
- [Dvorak-9 Scorer Documentation](README_dvorak9.md)  
- [Engram Scorer Documentation](README_engram.md)

### Research Background
- Dvorak, A. (1936). "Typewriter Keyboard"
- Dhakal et al. (2018). "Observations on Typing from 136 Million Keystrokes." CHI 2018.
- [Keyboard Layout Analyzer](http://patorjk.com/keyboard-layout-analyzer/)
- [Carpalx](http://mkweb.bcgsc.ca/carpalx/)

### Data Sources
- English bigram frequencies: Peter Norvig's analysis of Google Books data
- Typing performance data: 136M Keystrokes dataset
- Comfort ratings: Typing preference studies

## License

MIT License - See LICENSE file for details.

---

## Version History

**v2.0.0** - Unified Framework Release
- Standardized CLI across all scorers
- YAML configuration system
- Modular architecture with shared utilities
- Consistent output formats
- Comprehensive validation and error handling

**v1.x** - Original Individual Scorers
- Separate implementations for each scoring method
- Custom CLI and output for each scorer
- Limited code reuse between scorers