# Keyboard Layout Scorer
A comprehensive framework for evaluating keyboard layouts using precomputed scoring tables

Repository: https://github.com/binarybottle/keyboard_layout_scorers.git  
Author: Arno Klein (arnoklein.info)  
License: MIT License (see LICENSE)

This framework provides fast keyboard layout evaluation using precomputed scoring tables across multiple metrics. It uses a two-phase approach: first generate comprehensive scoring tables for all possible key-pair combinations, then use those tables for rapid layout scoring and comparison.

## Scoring Methods

The framework supports biomechanically-grounded scoring approaches by default, with experimental metrics available separately:

### Core metrics (default)
- Dvorak-7 scorer: Implementation of Dvorak's 7 theoretical typing principles
- Comfort scorer: Frequency-weighted key-pair comfort analysis based on Typing Preference Study data
- Comfort-key scorer: Frequency-weighted key comfort analysis based on Typing Preference Study data
- Engram scorer: Composite (key and key-pair) comfort score

### Experimental metrics (--experimental-metrics)
- Efficiency scorer: Physical finger travel distance analysis (oversimplifies biomechanics)
- Speed scorer: Typing time analysis based on empirical data (contains QWERTY practice bias)
- Dvorak-7 speed: Speed-weighted Dvorak-7 (also contains QWERTY bias)

Experimental metrics are disabled by default due to significant limitations:
  - Distance/efficiency metrics limitations:
    - Oversimplifies biomechanics: ignores lateral finger stretching vs. comfortable curling motions
    - Misses finger strength differences: treats pinky and index finger movements as equivalent
    - Uniform movement assumption: all finger movements treated as biomechanically equivalent
  - Time/speed metrics limitations:
    - Qwerty practice bias: empirical timing data reflects years of Qwerty-specific training
    - Unfair disadvantages: Alternative layouts map to less-practiced key combinations

## Installation

```bash
# Install dependencies
pip install pyyaml pandas numpy matplotlib

# Or with poetry
poetry install
```

## Quick Start

### Phase 1: Generate Scoring Tables

First, generate the precomputed scoring tables (already provided, language-independent):

```bash
# Generate individual scoring tables
python prep_keypair_engram8_scores.py
python prep_keypair_comfort_scores.py
python prep_keypair_dvorak7_scores.py
python prep_keypair_distance_scores.py --text-files corpus1.txt,corpus2.txt
python prep_keypair_time_scores.py --input-dir /path/to/csv/typing/data/

# Combine into unified scoring tables
python prep_scoring_tables.py --input-dir tables/ --verbose
```

### Phase 2: Score and Compare Layouts

Once tables are generated, score layouts quickly:

```bash
# Score using core biomechanical metrics (recommended)
python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Include experimental distance/time metrics (caution: limitations noted above)
python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --experimental-metrics

# Compare multiple layouts with core metrics (recommended)
python score_layouts.py --compare qwerty:"qwertyuiopasdfghjkl;zxcvbnm,./" dvorak:"',.pyfgcrlaoeuidhtns;qjkxbmwvz" colemak:"qwfpgjluy;arstdhneiozxcvbkm,./"

# Compare with experimental metrics (use with caution)
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --experimental-metrics

# Generate comparison visualizations (core metrics)
python compare_layouts.py --tables layout_scores.csv --output comparison

# Include experimental metrics in visualizations (caution)
python compare_layouts.py --metrics engram comfort efficiency_total speed_total --tables layout_scores.csv --output comparison_exp

# CSV output for automation (core metrics)
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output
```

## Architecture

```
keyboard_layout_scorers/
├── README.md                          # This file
│ 
├── # Table Generation (Phase 1)
├── prep_keypair_distance_scores.py    # Generate distance scoring table
├── prep_keypair_time_scores.py        # Generate time scoring table  
├── prep_keypair_comfort_scores.py     # Generate comfort scoring table
├── prep_keypair_dvorak7_scores.py     # Generate Dvorak-7 scoring table
├── prep_scoring_tables.py             # Combine tables with normalization
│
├── # Layout Scoring (Phase 2)
├── score_layouts.py                   # Main layout scoring tool
├── compare_layouts.py                 # Visualization and comparison
│
├── # Data Directories
├── tables/
│   ├── keypair_scores_detailed.csv    # Detailed unified scoring table
│   ├── keypair_scores_composite.csv   # Composite scoring table
│   ├── key_scores.csv                 # Individual key comfort scores
│   ├── keypair_distance_scores.csv    # Distance scoring data
│   ├── keypair_time_scores.csv        # Time scoring data
│   ├── keypair_comfort_scores.csv     # Comfort scoring data
│   ├── keypair_dvorak7_scores.csv     # Dvorak-7 scoring data
│   ├── keypair_engram8_scores.csv     # Engram-8 scoring data
│   └── keypair_*_*_scores.csv         # Individual criterion files│
│ 
├── input/                             # Input data files
│   ├── english-letter-pair-frequencies-google-ngrams.csv
│   ├── english-letter-frequencies-google-ngrams.csv
│   └── comfort_keypair_scores_24keys.csv
└──
```

## Workflow Details

### Phase 1: Table Generation

**Engram-8 Scores** (`prep_keypair_engram8_scores.py`):
- Implements 8 typing criteria from Typing Preference Study
- Generates both overall and individual criterion scores  
- Outputs: `keypair_engram8_scores.csv` + individual criterion files

**Comfort Scores** (`prep_keypair_comfort_scores.py`):
- Generates comfort scores for all key-pairs using rule-based approach
- Handles same-hand vs different-hand transitions
- Outputs: `keypair_comfort_scores.csv`

**Dvorak-7 Scores** (`prep_keypair_dvorak7_scores.py`):
- Implements all 7 Dvorak theoretical principles
- Generates both overall and individual criterion scores
- Outputs: `keypair_dvorak7_scores.csv` + individual criterion files

**Distance Scores** (`prep_keypair_distance_scores.py`):
- Computes theoretical distances for all 1024 Qwerty key-pairs
- Calculates distance components (setup, transition, return)
- Uses finger tracking with reset logic
- Outputs: `keypair_distance_scores.csv`
- Results inverted to become experimental efficiency metrics

**Time Scores** (`prep_keypair_time_scores.py`):
- Analyzes CSV typing data for key-to-key timing
- Calculates timing components (setup, transition, return)
- Outputs: `keypair_time_scores.csv`
- Results inverted to become experimental speed metrics

**Scoring Ranges:**
- Dvorak-7: Raw scores 0-7 (sum of 7 components), normalized to 0-1 
- Engram-8: Raw scores 0-8 (sum of 8 components), normalized to 0-1
- Universal normalization applied by prep_scoring_tables.py for cross-dataset comparability

**Table Unification** (`prep_scoring_tables.py`):
- Combines all individual score files
- Applies universal normalization ranges for cross-dataset comparability
- Creates `keypair_scores_detailed.csv`, `keypair_scores_composite.csv`, and `key_scores.csv`
- Handles both raw scores (0-7, 0-8 ranges) and normalized versions (0-1)

### Phase 2: Layout scoring

**Main scorer** (`score_layouts.py`):
- Fast lookup-based scoring using precomputed tables
- Supports frequency weighting using English (or other language) bigram data
- Multiple output formats: detailed, CSV, score-only
- Handles single layouts and multi-layout comparisons
- Default metrics: engram, comfort, comfort-key, dvorak7
- Experimental metrics: distance/time metrics with `--experimental-metrics`

**Ranking and visualization** (`compare_layouts.py`):
- Ranks layouts
- Creates parallel coordinates and heatmap plots  
- Supports multiple scoring tables

## Output Formats

### Detailed output (default)
Shows comprehensive results with scoring breakdowns:
```bash
# Core biomechanical metrics only
python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Include experimental metrics
python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ" --experimental-metrics
```

### CSV output for automation
```bash
# Minimal CSV with core metrics (layout_name,scorer,weighted_score,raw_score)
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output

# Include experimental metrics in CSV
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --experimental-metrics --csv-output

# Detailed CSV file
python score_layouts.py --compare layouts... --csv results.csv
```

### Visualization output
```bash
# Generate parallel coordinates and heatmap plots (core metrics)
python compare_layouts.py --tables layout_scores.csv --output comparison
# Creates: comparison_parallel.png and comparison_heatmap.png

# Include experimental metrics in visualization
python compare_layouts.py --metrics engram comfort efficiency_total speed_total --tables layout_scores.csv --output comparison_experimental
```

## Data Requirements

### Required files (Phase 1)
- `input/english-letter-pair-frequencies-google-ngrams.csv` - English bigram frequencies
- `input/english-letter-frequencies-google-ngrams.csv` - English letter frequencies  
- `input/comfort_keypair_scores_24keys.csv` - Base comfort scores for key-pairs
- Text corpus files (for preparing distance scoring table)
- CSV typing data directory (for preparing time scoring table)

### Generated files (Phase 1 output)
- `tables/keypair_scores_detailed.csv` - Unified scoring table with normalized scores
- `tables/key_scores.csv` - Individual key comfort scores
- Individual scorer tables (`tables/keypair_*_scores.csv`)

### File formats

**Frequency files:**
```csv
letter_pair,normalized_frequency
th,3.56
he,3.07
```

**Scoring tables:**
```csv
key_pair,distance_score,comfort_score,time_score,dvorak7_score
QW,45.2,0.7,150.3,0.6
AS,15.1,0.9,89.2,0.8
```

**Layout comparison output:**
```csv
layout_name,scorer,weighted_score,raw_score
qwerty,comfort,0.756234,0.742156
qwerty,engram,0.623451,0.618923
dvorak,comfort,0.834567,0.821234
```

## Advanced features

### Core vs experimental metrics
Select core biomechanical metrics or include experimental ones:

#### Core metrics (recommended)
```bash
# All core metrics (default behavior)
python score_layouts.py --letters "abc" --positions "ABC"

# Specific core metrics
python score_layouts.py --scorers engram,comfort --letters "abc" --positions "ABC"

# Single core metric
python score_layouts.py --scorer comfort-key --letters "abc" --positions "ABC"
```

#### Experimental metrics (use with caution)
```bash
# Include experimental distance/time metrics
python score_layouts.py --letters "abc" --positions "ABC" --experimental-metrics

# Specific experimental metric (requires --experimental-metrics)
python score_layouts.py --scorer efficiency --letters "abc" --positions "ABC" --experimental-metrics

# Mix core and experimental
python score_layouts.py --scorers speed,efficiency --letters "abc" --positions "ABC" --experimental-metrics
```

### Visualization options
```bash
# Core metrics visualization (recommended)
python compare_layouts.py --tables scores.csv --output comparison

# Specific core metrics
python compare_layouts.py --metrics engram comfort comfort-key dvorak7 --tables scores.csv --output core_comparison

# Include experimental metrics in visualization (caution)
python compare_layouts.py --metrics engram comfort efficiency_total speed_total --tables scores.csv --output experimental_comparison

# Verbose output with metric categorization
python compare_layouts.py --tables scores.csv --verbose
```

### Rankings and analysis
```bash
# Core metrics rankings
python compare_layouts.py --metrics engram comfort comfort-key dvorak7 --rankings core_rankings.csv --tables scores.csv

# Include experimental metrics in rankings (caution)
python compare_layouts.py --metrics engram comfort efficiency_total speed_total --rankings exp_rankings.csv --tables scores.csv

# Rank-based visualization coloring
python compare_layouts.py --metrics engram comfort dvorak7 --rankings rankings.csv --output ranked_comparison.png --tables scores.csv
```

### Automation-friendly output
```bash
# Get core metric scores for scripting (CSV format to stdout)
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output > core_results.csv

# Include experimental metrics for comprehensive data collection
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --experimental-metrics --csv-output > all_results.csv

# Process results with visualizations
python compare_layouts.py --tables core_results.csv --output core_analysis
python compare_layouts.py --tables all_results.csv --metrics engram comfort efficiency_total --output mixed_analysis
```