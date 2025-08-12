# Keyboard Layout Scorer
A comprehensive framework for evaluating keyboard layouts using precomputed scoring tables

**Repository**: https://github.com/binarybottle/keyboard_layout_scorers.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

This framework provides fast keyboard layout evaluation using precomputed scoring tables across multiple metrics. It uses a two-phase approach: first generate comprehensive scoring tables for all possible key-pair combinations, then use those tables for rapid layout scoring and comparison.

## Scoring Methods

The framework supports four main scoring approaches:

- **Distance Scorer**: Physical finger travel distance analysis with per-finger distance tracking
- **Time Scorer**: Typing time analysis based on empirical data from CSV typing datasets
- **Comfort Scorer**: Key comfort analysis based on finger ergonomics and key positions
- **Dvorak-7 Scorer**: Implementation of Dvorak's 9 theoretical typing principles with individual criterion breakdown

## Installation

```bash
# Install dependencies
pip install pyyaml pandas numpy matplotlib

# Or with poetry
poetry install
```

## Quick Start

### Phase 1: Generate Scoring Tables

First, generate the precomputed scoring tables (this only needs to be done once):

```bash
# Generate individual scoring tables
python prep_keypair_distance_scores.py --text-files corpus1.txt,corpus2.txt
python prep_keypair_time_scores.py --input-dir /path/to/csv/typing/data/
python prep_keypair_comfort_scores.py
python prep_keypair_dvorak7_scores.py

# Combine into unified scoring tables
python prep_scoring_tables.py --input-dir tables/ --verbose
```

### Phase 2: Score and Compare Layouts

Once tables are generated, score layouts quickly:

```bash
# Score a single layout using all available methods
python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"

# Compare multiple layouts
python score_layouts.py --compare qwerty:"qwertyuiopasdfghjkl;zxcvbnm,./" dvorak:"',.pyfgcrlaoeuidhtns;qjkxbmwvz" colemak:"qwfpgjluy;arstdhneiozxcvbkm,./"

# Generate comparison visualizations
python compare_layouts.py --tables layout_scores.csv --output comparison

# CSV output for automation
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output
```

### Automated Workflow

Generate ready-to-run commands for standard layout comparison:

```bash
# Generate workflow commands
python generate_command.py

# Run the complete workflow
./output/compare_standard_layouts.sh
```

## Architecture

```
keyboard_layout_scorers/
├── README.md                          # This file
├── 
├── # Table Generation (Phase 1)
├── prep_keypair_distance_scores.py    # Generate distance scoring table
├── prep_keypair_time_scores.py        # Generate time scoring table  
├── prep_keypair_comfort_scores.py     # Generate comfort scoring table
├── prep_keypair_dvorak7_scores.py     # Generate Dvorak-7 scoring table
├── prep_scoring_tables.py             # Combine tables with normalization
├──
├── # Layout Scoring (Phase 2)
├── score_layouts.py                   # Main layout scoring tool
├── compare_layouts.py                 # Visualization and comparison
├── generate_command.py                # Workflow automation
├──
├── # Data Directories
├── tables/                            # Generated scoring tables
│   ├── keypair_scores.csv             # Unified scoring table
│   ├── key_scores.csv                 # Individual key comfort scores
│   ├── keypair_distance_scores.csv    # Distance scoring data
│   ├── keypair_time_scores.csv        # Time scoring data
│   ├── keypair_comfort_scores.csv     # Comfort scoring data
│   └── keypair_dvorak7_scores.csv     # Dvorak-7 scoring data
├──
├── input/                             # Input data files
│   ├── english-letter-pair-frequencies-google-ngrams.csv
│   ├── english-letter-frequencies-google-ngrams.csv
│   └── comfort_keypair_scores_24keys.csv
└──
```

## Workflow Details

### Phase 1: Table Generation

**Distance Scores** (`prep_keypair_distance_scores.py`):
- Computes theoretical distances for all 1024 QWERTY key-pairs
- Processes text files to find real-world usage patterns
- Uses finger tracking with proper reset logic
- Outputs: `keypair_distance_scores.csv`

**Time Scores** (`prep_keypair_time_scores.py`):
- Analyzes CSV typing data for key-to-key timing
- Uses comprehensive fallback strategy for missing data
- Calculates timing components (key1_time + transition_time)
- Outputs: `keypair_time_scores.csv`

**Comfort Scores** (`prep_keypair_comfort_scores.py`):
- Generates comfort scores for all key-pairs using rule-based approach
- Handles same-hand vs different-hand transitions
- Outputs: `keypair_comfort_scores.csv`

**Dvorak-7 Scores** (`prep_keypair_dvorak7_scores.py`):
- Implements all 9 Dvorak theoretical principles
- Generates both overall and individual criterion scores
- Outputs: `keypair_dvorak7_scores.csv` + individual criterion files

**Table Unification** (`prep_scoring_tables.py`):
- Combines all individual score files
- Applies smart normalization with distribution detection
- Creates unified `keypair_scores.csv` and `key_scores.csv`

### Phase 2: Layout Scoring

**Main Scorer** (`score_layouts.py`):
- Fast lookup-based scoring using precomputed tables
- Supports frequency weighting using English bigram data
- Multiple output formats: detailed, CSV, score-only
- Handles single layouts and multi-layout comparisons

**Visualization** (`compare_layouts.py`):
- Creates parallel coordinates plots
- Generates heatmap visualizations  
- Sorts layouts by performance
- Supports multiple scoring tables

## Output Formats

### Detailed Output (Default)
Shows comprehensive results with scoring breakdowns:
```bash
python score_layouts.py --letters "etaoinshrlcu" --positions "FDESGJWXRTYZ"
```

### CSV Output for Automation
```bash
# Minimal CSV (layout_name,scorer,weighted_score,raw_score)
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output

# Detailed CSV file
python score_layouts.py --compare layouts... --csv results.csv
```

### Visualization Output
```bash
# Generate parallel coordinates and heatmap plots
python compare_layouts.py --tables layout_scores.csv --output comparison
# Creates: comparison_parallel.png and comparison_heatmap.png
```

## Data Requirements

### Required Files (Phase 1)
- `input/english-letter-pair-frequencies-google-ngrams.csv` - English bigram frequencies
- `input/english-letter-frequencies-google-ngrams.csv` - English letter frequencies  
- `input/comfort_keypair_scores_24keys.csv` - Base comfort scores for key-pairs
- Text corpus files (for preparing distance scoring table)
- CSV typing data directory (for preparing time scoring table)

### Generated Files (Phase 1 Output)
- `tables/keypair_scores.csv` - Unified scoring table with normalized scores
- `tables/key_scores.csv` - Individual key comfort scores
- Individual scorer tables (`tables/keypair_*_scores.csv`)

### File Formats

**Frequency Files:**
```csv
letter_pair,normalized_frequency
th,3.56
he,3.07
```

**Scoring Tables:**
```csv
key_pair,distance_score,comfort_score,time_score,dvorak7_score
QW,45.2,0.7,150.3,0.6
AS,15.1,0.9,89.2,0.8
```

**Layout Comparison Output:**
```csv
layout_name,scorer,weighted_score,raw_score
qwerty,distance,0.756234,0.742156
qwerty,comfort,0.623451,0.618923
dvorak,distance,0.834567,0.821234
```

## Advanced Features

### Frequency Weighting
Uses English bigram frequencies to weight scoring by real-world usage patterns:
```bash
# Frequency-weighted scoring (default)
python score_layouts.py --letters "abc" --positions "ABC"

# Raw unweighted scoring  
python score_layouts.py --letters "abc" --positions "ABC" --raw
```

### Multiple Scoring Methods
Select specific scorers or use all available:
```bash
# Single scorer
python score_layouts.py --scorer distance --letters "abc" --positions "ABC"

# Multiple specific scorers
python score_layouts.py --scorers distance,comfort --letters "abc" --positions "ABC"

# All available scorers (default)
python score_layouts.py --letters "abc" --positions "ABC"
```

### Visualization Options
```bash
# Basic visualization
python compare_layouts.py --tables scores.csv --output comparison

# Raw scores instead of weighted
python compare_layouts.py --tables scores.csv --use-raw --output comparison_raw

# Verbose output with statistics
python compare_layouts.py --tables scores.csv --verbose
```

## Standard Layouts Included

The framework includes analysis for 13 standard keyboard layouts:

- **QWERTY** - `qwertyuiopasdfghjkl;zxcvbnm,./['`
- **Dvorak** - `',.pyfgcrlaoeuidhtns;qjkxbmwvz['`
- **Colemak** - `qwfpgjluy;arstdhneiozxcvbkm,./['`
- **Colemak-DH** - `qwfpbjluy;arstgmneiozxcdvkh,./['`
- **Workman** - `qdrwbjfup;ashtgyneoizxmcvkl,./['`
- **Norman** - `qwdfkjurl;asetgyniohzxcvbpm,./['`
- **Halmak 2.2** - `wlrbz;qudjshnt,.aeoifmvc/gpxky['`
- **Asset** - `qwfgjypul;asetdhniorzxcvbkm,./['`
- **MTGAP 2.0** - `,fhdkjcul.oantgmseriqxbpzyw'v;['`
- **QGMLWB** - `qgmlwbyuv;dstnriaeohzxcfjkp,./['`
- **Capewell-Dvorak** - `',.pyqfgrkoaeiudhtnszxcvjlmwb;['`
- **Klausler** - `k,uypwlmfcoaeidrnthsq.';zxvgbj['`
- **Hieamtsrn** - `byou'kdclphiea,mtsrnx-".?wgfjzqv`

## Example Usage

### Complete Standard Layout Analysis
```bash
# Generate workflow for standard layouts
python generate_command.py

# Run complete analysis (generates tables if needed, then compares layouts)
./output/compare_standard_layouts.sh

# Results: layout_comparison_parallel.png, layout_comparison_heatmap.png, layout_scores.csv
```

### Custom Layout Analysis
```bash
# Define custom layout
python score_layouts.py --letters "etaoinshrdlcu" --positions "QWERTYUIOPASD" --verbose

# Compare custom vs standard
python score_layouts.py --compare custom:"etaoinshrdlcu" qwerty:"qwertyuiopasdf"
```

### Automation-Friendly Output
```bash
# Get scores for scripting (CSV format to stdout)
python score_layouts.py --compare qwerty:"qwertyuiop" dvorak:"',.pyfgcrl" --csv-output > results.csv

# Process results
python compare_layouts.py --tables results.csv --output analysis
```
