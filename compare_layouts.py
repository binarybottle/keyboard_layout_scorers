"""
Keyboard layout comparison

See README for information.

(c) 2021-2025 Arno Klein (arnoklein.info), MIT license

"""
import os
import subprocess
import pandas as pd
import requests
from io import StringIO
import tempfile
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# DEBUG: Set to a number to limit text length for faster testing, or None for full text
DEBUG_TEXT_LIMIT = None  # Change to None for full text processing
DEBUG_TIMEOUT = 600  # Increase timeout for debugging

# Create output directories
output_dirs = {
    'csv': 'output/csv',
    'md': 'output/md', 
    'png': 'output/png'
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

#===============================================================================
# Layout information
#===============================================================================
engram     = ['B','Y','O','U', 'C','I','E','A', 'G','X','J','K', 'L','D','W','V', 'H','T','S','N', 'R','M','F','P', "'",',','-', '"','.','?', 'Z','Q']
halmak     = ['W','L','R','B', 'S','H','N','T', 'F','M','V','C', 'Q','U','D','J', 'A','E','O','I', 'P','X','K','Y', 'Z',',','/', ';','.','G', '[',"'"]
hieamtsrn  = ['B','Y','O','U', 'H','I','E','A', 'X','-','"','.', 'D','C','L','P', 'T','S','R','N', 'G','F','J','Z', "'",',','?', 'K','M','W', 'Q','V']
norman     = ['Q','W','D','F', 'A','S','E','T', 'Z','X','C','V', 'U','R','L',';', 'N','I','O','H', 'M',',','.','/', 'K','G','B', 'J','Y','P', '[',"'"]
workman    = ['Q','D','R','W', 'A','S','H','T', 'Z','X','M','C', 'F','U','P',';', 'N','E','O','I', 'L',',','.','/', 'B','G','V', 'J','Y','K', '[',"'"]
mtgap2     = [',','F','H','D', 'O','A','N','T', 'Q','X','B','P', 'C','U','L','.', 'S','E','R','I', 'W',"'",'V',';', "K",'G','Z', 'J','M','Y', '[','/'] 
qgmlwb     = ['Q','G','M','L', 'D','S','T','N', 'Z','X','C','F', 'Y','U','V',';', 'A','E','O','H', 'P',',','.','/', 'W','R','J', 'B','I','K', '[',"'"]
colemakmod = ['Q','W','F','P', 'A','R','S','T', 'Z','X','C','D', 'L','U','Y',';', 'N','E','I','O', 'H',',','.','/', 'B','J','G', 'K','V','M', '[',"'"]
colemak    = ['Q','W','F','P', 'A','R','S','T', 'Z','X','C','V', 'L','U','Y',';', 'N','E','I','O', 'M',',','.','/', 'G','D','B', 'J','H','K', '[',"'"]
asset      = ['Q','W','J','F', 'A','S','E','T', 'Z','X','C','V', 'P','U','L',';', 'N','I','O','R', 'M',',','.','/', 'G','Y','D', 'H','B','K', '[',"'"] 
capewell   = ["'",',','.','P', 'O','A','E','I', 'Z','X','C','V', 'F','G','R','K', 'H','T','N','S', 'M','W','B',';', 'Y','Q','U', 'D','J','L', '/','-']
klausler   = ['K',',','U','Y', 'O','A','E','I', 'Q','.',"'",';', 'L','M','F','C', 'N','T','H','S', 'V','G','B','J', 'P','D','Z', 'W','R','X', '/',"-"]
dvorak     = ["'",',','.','P', 'A','O','E','U', ';','Q','J','K', 'G','C','R','L', 'H','T','N','S', 'M','W','V','Z', 'Y','F','I', 'D','X','B', '/','-']
qwerty     = ['Q','W','E','R', 'A','S','D','F', 'Z','X','C','V', 'U','I','O','P', 'J','K','L',';', 'M',',','.','/', 'T','G','B', 'Y','H','N', '[',"'"]

layout_names = ['Engram','Halmak','Hieamtsrn','Norman','Workman','MTGap 2.0','QGMLWB',
                'Colemak Mod-DH','Colemak','Asset','Capewell-Dvorak','Klausler','Dvorak','QWERTY']
layout_abbrs = ['Engram','Halmak','Hieamtsrn','Norman','Workman','MTGap','QGMLWB',
                'ColemakMod','Colemak','Asset','CapewellDvorak','Klausler','Dvorak','QWERTY']
layout_letters = [engram, halmak, hieamtsrn, norman, workman, mtgap2, qgmlwb,
                  colemakmod, colemak, asset, capewell, klausler, dvorak, qwerty]

layout_names_KLA = ['Engram','Halmak','Hieamtsrn','Norman','Workman','MTGap 2.0','QGMLWB',
                    'Colemak Mod-DH','Colemak','Asset','Capewell-Dvorak','Klausler','Dvorak']
layout_abbrs_KLA = ['Engram','Halmak','Hieamtsrn','Norman','Workman','MTGap','QGMLWB',
                    'ColemakMod','Colemak','Asset','CapewellDvorak','Klausler','Dvorak']
layout_letters_KLA = [engram, halmak, hieamtsrn, norman, workman, mtgap2, qgmlwb, 
                      colemakmod, colemak, asset, capewell, klausler, dvorak]

# Define standard QWERTY positions (32 most common positions)
# This maps the layout array indices to QWERTY key positions
# 
#    1  2  3  4   25  28   13 14 15 16   31
#    5  6  7  8   26  29   17 18 19 20   32
#    9 10 11 12   27  30   21 22 23 24 

STANDARD_POSITIONS = [
    'Q', 'W', 'E', 'R',  # Top row left
    'A', 'S', 'D', 'F',  # Home row left  
    'Z', 'X', 'C', 'V',  # Bottom row left
    'U', 'I', 'O', 'P',  # Top row right
    'J', 'K', 'L', ';',  # Home row right
    'M', ',', '.', '/',  # Bottom row right
    'T', 'G', 'B',       # Center columns
    'Y', 'H', 'N',       # Center columns
    '[', "'"             # Additional keys
]

#===============================================================================
# Text data
#===============================================================================
text_names = ["Alice in Wonderland (Ch.1)",
              "Memento screenplay",
              "100,000 tweets (Sentiment Classification)", 
              "20,000 tweets (Gender classifier)", 
              "Manually Annotated Sub-Corpus tweets",
              "Manually Annotated Sub-Corpus spoken transcripts",
              "Corpus of Contemporary American English blog samples",
              "Shai Coleman iweb corpus 1/6",
              "Ian Douglas monkey test",
              "Ian Douglas coder test",
              "Tower of Hanoi (programming languages A-Z, Rosetta Code)"]

text_abbrs = ["Alice", "Memento", "Tweets_100K", "Tweets_20K", "Tweets_MASC", 
              "Spoken_MASC", "COCA_blogs", "iweb", "Monkey", "Coder", "Rosetta"]
text_names.append("Google n-grams bigram frequencies")
text_abbrs.append("Bigrams")

text_indices = range(len(text_names))
text_names_copy = []
text_abbrs_copy = []
for text_index in text_indices:
    text_names_copy.append(text_names[text_index])
    text_abbrs_copy.append(text_abbrs[text_index])
text_names = text_names_copy
text_abbrs = text_abbrs_copy    

#===============================================================================
# Keyboard Layout Analyzer data
#===============================================================================
# Import the KLA data
try:
    from KeyboardLayoutAnalyzer_data import *
    print("Successfully imported KLA data")
except ImportError:
    print("Warning: Could not import KLA data from KeyboardLayoutAnalyzer_data.py")

finger_names = ["left little", "left ring", "left middle", "left index",
                "right index", "right middle", "right ring", "right little", "total"]
measure_abbrs = ["distance", "taps", "samefinger"]

#===============================================================================
# Scoring functions
#===============================================================================
def load_local_texts(text_abbrs, base_dir="../text_data"):
    """Load text files from local directory."""
    texts = {}
    
    # Map text abbreviations to likely filenames
    filename_map = {
        "Alice": "AliceInWonderland_Ch1.txt",
        "Memento": "Memento_screenplay.txt", 
        "Tweets_100K": "training.1600000.processed.noemoticon_1st100000tweets.txt",
        "Tweets_20K": "gender-classifier-20000tweets.txt",
        "Tweets_MASC": "MASC_tweets_cleaned.txt",
        "Spoken_MASC": "MASC_spoken_transcripts_of_phone_face2face.txt",
        "COCA_blogs": "COCA_corpusdata.org_sample_text_blog_cleaned.txt",
        "iweb": "iweb-corpus-cleaned-1.txt",
        "Monkey": "monkey0-7_IanDouglas.txt",
        "Coder": "coder0-7_IanDouglas.txt",
        "Rosetta": "rosettacode.org_TowersOfHanoi_AtoZ.txt"
    }
    
    for abbr in text_abbrs:
        filename = filename_map.get(abbr, f"{abbr}.txt")
        filepath = os.path.join(base_dir, filename)
        
        try:
            print(f"Loading {abbr} from {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                texts[abbr] = f.read()
            print(f"  Loaded {len(texts[abbr]):,} characters")
        except Exception as e:
            print(f"  Error loading {abbr}: {e}")
            texts[abbr] = ""  # Use empty string as fallback
    
    return texts

def create_layout_mapping(layout_chars):
    """Create mapping from characters to their QWERTY positions."""
    char_to_pos = {}
    
    for i, char in enumerate(layout_chars):
        if i < len(STANDARD_POSITIONS):
            # Find where this character appears in the layout
            # char_to_pos[char] = STANDARD_POSITIONS[i]
            
            # Actually, we need to reverse this:
            # The layout tells us what character is at each position
            # We need to know what position each character is at
            char_to_pos[char.lower()] = STANDARD_POSITIONS[i]
    
    return char_to_pos

def run_score_layout(items, positions, config_path="config.yaml"): 
    """Run score_layout.py and parse CSV output."""
    try:
        cmd = [
            'poetry', 'run', 'python3', 'score_layout.py',
            '--items', items,
            '--positions', positions,
            '--config', config_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=DEBUG_TIMEOUT,
                               cwd="../optimize_layouts")
                
        if result.returncode != 0:
            print(f"Error running score_layout.py: {result.stderr}")
            return None
        
        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        csv_start = -1
        
        for i, line in enumerate(lines):
            if line.startswith('total_score,item_score,item_pair_score'):
                csv_start = i
                break
        
        if csv_start == -1:
            print("Could not find CSV output in score_layout.py")
            return None
        
        csv_data = '\n'.join(lines[csv_start:csv_start+2])  # Header + data
        df = pd.read_csv(StringIO(csv_data))
        
        if len(df) > 0:
            return {
                'total_score': df.iloc[0]['total_score'],
                'item_score': df.iloc[0]['item_score'], 
                'item_pair_score': df.iloc[0]['item_pair_score']
            }
    except Exception as e:
        print(f"Error running score_layout.py: {e}")
        return None

def run_dvorak9_scorer(items, positions, text, weights_csv="dvorak9_weights.csv"):
    """Run dvorak9_scorer.py and parse CSV output."""
    try:
        # Write text to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            text_file = f.name
        
        cmd = [
            'poetry', 'run', 'python3', 'dvorak9_scorer.py',
            '--items', items,
            '--positions', positions,
            '--text-file', text_file,
            '--weights-csv', weights_csv,
            '--ten-scores'  # Remove --csv since --ten-scores gives space-separated output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=DEBUG_TIMEOUT,
                               cwd="../dvorak9-scorer")
        
        # Clean up temp file
        os.unlink(text_file)
        
        if result.returncode != 0:
            print(f"Error running dvorak9_scorer.py: {result.stderr}")
            return None
        
        # Parse space-separated output
        lines = result.stdout.strip().split('\n')
        
        # Find the line with scores (skip informational lines)
        scores_line = None
        for line in lines:
            line = line.strip()
            # Look for line that starts with a number (positive or negative)
            if line and (line[0].isdigit() or line[0] == '-'):
                scores_line = line
                break
        
        if not scores_line:
            print("Could not find scores line in dvorak9_scorer.py output")
            return None
        
        # Parse space-separated scores
        try:
            score_values = [float(x) for x in scores_line.split()]
        except Exception as e:
            print(f"    Failed to parse scores: {e}")
            return None
        
        # Map scores to metric names (based on dvorak9 script output order)
        metric_names = [
            'total_weighted_score', # Total weighted score (first value)
            'individual_hands',
            'individual_fingers', 
            'individual_skip_fingers',
            'individual_dont_cross_home',
            'individual_same_row',
            'individual_home_row',
            'individual_columns',
            'individual_strum',
            'individual_strong_fingers'
        ]
        
        # Create scores dictionary
        scores = {}
        for i, metric in enumerate(metric_names):
            if i < len(score_values):
                scores[metric] = score_values[i]
        
        return scores
        
    except Exception as e:
        print(f"Error running dvorak9_scorer.py: {e}")
        return None
          
def clean_text_for_scoring(text):
    """Clean text for scoring."""
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_common_characters(layout_letters):
    """Get the set of characters that appear in all layouts."""
    all_chars = set()
    for layout in layout_letters:
        all_chars.update(char.lower() for char in layout)
    
    # Sort by frequency/importance (rough approximation)
    char_priority = "etaoinshrdlcumwfgypbvkjxqz.,';/[]"
    
    common_chars = []
    for char in char_priority:
        if char in all_chars:
            common_chars.append(char)
    
    # Add any remaining characters
    for char in sorted(all_chars):
        if char not in common_chars:
            common_chars.append(char)
    
    return ''.join(common_chars)

#===============================================================================
# Display functions
#===============================================================================
def save_dataframe_as_markdown(df, csv_filename, title=None, divide_by=1, display_int=True):
    """Save DataFrame as markdown table."""
    # Convert CSV path to MD path
    base_filename = os.path.basename(csv_filename).replace('.csv', '.md')
    md_filename = os.path.join('output/md', base_filename)
    
    with open(md_filename, 'w') as f:
        if title:
            f.write(f"# {title}\n\n")
        
        # Write header
        f.write("| Layout |")
        for col in df.columns:
            f.write(f" {col} |")
        f.write("\n")
        
        # Write separator
        f.write("| --- |")
        for col in df.columns:
            f.write(" --- |")
        f.write("\n")
        
        # Write data rows
        for layout in df.index:
            f.write(f"| {layout} |")
            for col in df.columns:
                score = df.loc[layout, col]
                if pd.isna(score):
                    formatted_score = "N/A"
                else:
                    if divide_by > 1:
                        score = score / divide_by
                    
                    if display_int:
                        formatted_score = f"{score:.0f}"
                    else:
                        formatted_score = f"{score:.2f}"
                
                f.write(f" {formatted_score} |")
            f.write("\n")
    
    print(f"Saved: {md_filename}")

def save_dataframe_as_heatmap(df, csv_filename, title=None, divide_by=1):
    """Save DataFrame as heatmap with viridis colormap."""
    # Convert CSV path to PNG path
    base_filename = os.path.basename(csv_filename).replace('.csv', '.png')
    png_filename = os.path.join('output/png', base_filename)
    
    # Convert to numpy array and apply divide_by
    data = df.values.copy().astype(float)
    if divide_by > 1:
        data = data / divide_by
    
    # Take absolute values and normalize by column max
    data_abs = np.abs(data)
    columns_max = np.nanmax(data_abs, axis=0)
    # Avoid division by zero
    columns_max[columns_max == 0] = 1
    data_normalized = data_abs / columns_max
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_normalized, 
                xticklabels=df.columns,
                yticklabels=df.index,
                cmap='viridis',
                cbar=True,
                fmt='.2f')
    
    if title:
        plt.title(title)
    
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_filename}")

def generate_bigram_text(csv_file='letter_pair_frequencies_english.csv', output_file='output/bigrams_text_data.txt'):
    """Generate synthetic text from bigram frequencies."""
    print(f"Loading bigram frequencies from {csv_file}...")
    
    try:
        bigram_freq = pd.read_csv(csv_file)
        print(f"Loaded {len(bigram_freq)} bigrams")
    except Exception as e:
        print(f"Error loading bigram frequencies: {e}")
        return ""
    
    # Get common characters for filtering
    common_chars = set(get_common_characters(layout_letters))
    
    # Generate text
    bigram_text_parts = []
    total_bigrams_processed = 0
    
    for _, row in bigram_freq.iterrows():
        bigram = str(row['item_pair']).lower()  # Changed from 'bigram' to 'item_pair'
        frequency = int(row['score'])           # Changed from 'frequency' to 'score'
        
        # Filter: only include bigrams where both characters are in common_chars
        if len(bigram) == 2 and all(char in common_chars for char in bigram):
            repetitions = frequency // 100000
            if repetitions > 0:
                # Add space-flanked bigrams
                bigram_with_spaces = f" {bigram} "
                bigram_text_parts.extend([bigram_with_spaces] * repetitions)
                total_bigrams_processed += repetitions
    
    # Join all parts
    bigram_text = "".join(bigram_text_parts)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(bigram_text)
    
    print(f"Generated bigram text with {total_bigrams_processed:,} bigram instances")
    print(f"Saved: {output_file}")
    
    return bigram_text

#===============================================================================
# Engram scores
#===============================================================================
# Generate bigram frequency text
bigram_text_content = generate_bigram_text()

print("Loading text files from local directory...")
texts = load_local_texts(text_abbrs[:-1])  # Load all except Bigrams

# Add the generated bigram text
texts['Bigrams'] = bigram_text_content

# Get common character set for scoring
common_items = get_common_characters(layout_letters)
print(f"Using {len(common_items)} common characters: {common_items}")

# Initialize results storage
score_layout_results = {}  # [layout][text] = score
dvorak9_results = {}       # [layout][text][metric] = score

print("\nRunning scoring comparisons...")

# Process each layout
for layout_idx, (layout_name, layout_chars) in enumerate(zip(layout_abbrs, layout_letters)):
    print(f"\nProcessing {layout_name} ({layout_idx+1}/{len(layout_letters)})...")
    
    # Create character-to-position mapping for this layout
    char_to_pos = create_layout_mapping(layout_chars)
    
    # Build positions string for characters in common_items
    positions = ""
    items_filtered = ""
    for char in common_items:
        if char in char_to_pos:
            items_filtered += char
            positions += char_to_pos[char]
    
    print(f"  Using {len(items_filtered)} characters: {items_filtered[:50]}...")
    
    score_layout_results[layout_name] = {}
    dvorak9_results[layout_name] = {}
    
    # Test with a short sample first
    sample_text = "the quick brown fox jumps over the lazy dog"
    
    # Run score_layout.py (doesn't need text)
    print(f"  Running score_layout.py...")
    score_result = run_score_layout(items_filtered, positions)

    if score_result:
        # Store single score since score_layout doesn't use text
        score_layout_results[layout_name] = score_result['total_score']
    else:
        print(f"  Failed to run score_layout.py for {layout_name}")
        score_layout_results[layout_name] = None

    # Run dvorak9_scorer.py for each text
    for text_idx, (text_abbr, text_content) in enumerate(zip(text_abbrs, texts.values())):
        print(f"  Running dvorak9_scorer.py with {text_abbr}...")
        
        # Clean text
        clean_text = clean_text_for_scoring(text_content)
        
        # DEBUG: Optionally limit text length
        if DEBUG_TEXT_LIMIT and len(clean_text) > DEBUG_TEXT_LIMIT:
            clean_text = clean_text[:DEBUG_TEXT_LIMIT]
            print(f"    Limited to {DEBUG_TEXT_LIMIT} chars for debugging")
        
        if not clean_text:
            print(f"    No valid text for {text_abbr}")
            dvorak9_results[layout_name][text_abbr] = {}
            continue
        
        dvorak_result = run_dvorak9_scorer(items_filtered, positions, clean_text)
        
        if dvorak_result:
            dvorak9_results[layout_name][text_abbr] = dvorak_result
        else:
            print(f"    Failed to run dvorak9_scorer.py for {layout_name} with {text_abbr}")
            dvorak9_results[layout_name][text_abbr] = {}

print("\nGenerating CSV tables...")

# Create score_layout.py results table
score_layout_df = pd.DataFrame(list(score_layout_results.items()), columns=['Layout', 'score'])
score_layout_df.set_index('Layout', inplace=True)
csv_file = 'output/csv/score_layout_results.csv'
score_layout_df.to_csv(csv_file)
print(f"Saved: {csv_file}")
save_dataframe_as_markdown(score_layout_df, csv_file, "Score Layout Results", divide_by=1, display_int=False)
save_dataframe_as_heatmap(score_layout_df, csv_file, "Score Layout Results", divide_by=1)

#===============================================================================
# Dvorak-9 scores
#===============================================================================
# Create dvorak9_scorer.py results tables
dvorak9_metrics = [
    'total_weighted_score',  # Total weighted score
    'individual_hands',
    'individual_fingers', 
    'individual_skip_fingers',
    'individual_dont_cross_home',
    'individual_same_row',
    'individual_home_row',
    'individual_columns',
    'individual_strum',
    'individual_strong_fingers'
]

for metric in dvorak9_metrics:
    # Create table for this metric
    metric_data = {}
    
    for layout_name in layout_abbrs:
        metric_data[layout_name] = {}
        for text_abbr in text_abbrs:
            if (layout_name in dvorak9_results and 
                text_abbr in dvorak9_results[layout_name] and
                metric in dvorak9_results[layout_name][text_abbr]):
                metric_data[layout_name][text_abbr] = dvorak9_results[layout_name][text_abbr][metric]
            else:
                metric_data[layout_name][text_abbr] = None
    
    metric_df = pd.DataFrame(metric_data).T  # Transpose so layouts are rows
    metric_df.index.name = 'Layout'
    
    # Clean up metric name for filename
    filename = f"output/csv/dvorak9_{metric.replace('individual_', '')}.csv"
    metric_df.to_csv(filename)
    print(f"Saved: {filename}")
    metric_title = metric.replace('individual_', '').replace('_', ' ').title()
    save_dataframe_as_markdown(metric_df, filename, f"Dvorak9 {metric_title}", divide_by=1, display_int=False)
    save_dataframe_as_heatmap(metric_df, filename, f"Dvorak9 {metric_title}", divide_by=1)

print(f"\nGenerated CSV tables:")
print(f"- output/score_layout_results.csv (1 table)")
print(f"- output/dvorak9_*.csv (10 tables)")
print(f"\nTables have layouts as rows and text sources as columns.")

print("\nProcessing KLA measures...")

#===============================================================================
# Keyboard Layout Analyzer scores
#===============================================================================
# Map layout names between the main script and KLA data
kla_layout_mapping = {
    'Engram': 'Engram2',  # Using Engram2 as default, could also use Engram3
    'Halmak': 'Halmak',
    'Hieamtsrn': 'Hieamtsrn', 
    'Norman': 'Norman',
    'Workman': 'Workman',
    'MTGap': 'MTGap',
    'QGMLWB': 'QGMLWB',
    'ColemakMod': 'ColemakModDH',
    'Colemak': 'Colemak',
    'Asset': 'Asset',
    'CapewellDvorak': 'CapewellDvorak',
    'Klausler': 'Klausler',
    'Dvorak': 'Dvorak'
}

# Map text names between main script and KLA data
kla_text_mapping = {
    'Alice': 'Alice',
    'Memento': 'Memento', 
    'Tweets_100K': 'Tweets_100K',
    'Tweets_20K': 'Tweets_20K',
    'Tweets_MASC': 'Tweets_MASC',
    'Spoken_MASC': 'Spoken_MASC',
    'COCA_blogs': 'COCA_blogs',
    'iweb': 'iweb',
    'Monkey': 'Monkey',
    'Coder': 'Coder',
    'Rosetta': 'Rosetta'
}

# Initialize matrices for KLA metrics
import numpy as np
empty_matrix = np.zeros((len(layout_abbrs_KLA), len(text_abbrs)))

# Finger distances (cm)  
total_finger_distances = empty_matrix.copy()
left_little_finger_distances = empty_matrix.copy()
right_little_finger_distances = empty_matrix.copy()
left_index_finger_distances = empty_matrix.copy()
right_index_finger_distances = empty_matrix.copy()
# Finger taps  
left_little_finger_taps = empty_matrix.copy()
right_little_finger_taps = empty_matrix.copy()
same_finger_different_key_taps = empty_matrix.copy()  
# Balance between left and right hands (percent, left side negative)
hand_balances = empty_matrix.copy()

# Extract KLA data
for ilayout, layout_abbr in enumerate(layout_abbrs_KLA):
    kla_layout_name = kla_layout_mapping.get(layout_abbr, layout_abbr)
    
    for itext, text_abbr in enumerate(text_abbrs):
        kla_text_name = kla_text_mapping.get(text_abbr, text_abbr)
        
        try:
            # Dynamically get the variables from the imported KLA data
            distances_var = f'{kla_text_name}_{measure_abbrs[0]}_{kla_layout_name}'
            taps_var = f'{kla_text_name}_{measure_abbrs[1]}_{kla_layout_name}'
            samefingers_var = f'{kla_text_name}_{measure_abbrs[2]}_{kla_layout_name}'
            
            distances = globals()[distances_var]
            taps = globals()[taps_var]
            samefingers = globals()[samefingers_var]
            
            total_finger_distances[ilayout, itext] = np.sum(distances)
            left_little_finger_distances[ilayout, itext] = distances[0]
            right_little_finger_distances[ilayout, itext] = distances[7]
            left_index_finger_distances[ilayout, itext] = distances[3]
            right_index_finger_distances[ilayout, itext] = distances[4]
            left_little_finger_taps[ilayout, itext] = taps[0]
            right_little_finger_taps[ilayout, itext] = taps[7]
            same_finger_different_key_taps[ilayout, itext] = np.sum(samefingers)
            hand_balances[ilayout, itext] = 100 * (np.sum(taps[4:8]) - np.sum(taps[0:4])) / np.sum(taps[0:8])
            
        except KeyError:
            print(f"Warning: Could not find KLA data for {layout_abbr} with {text_abbr}")
            # Fill with NaN for missing data
            total_finger_distances[ilayout, itext] = np.nan
            left_little_finger_distances[ilayout, itext] = np.nan
            right_little_finger_distances[ilayout, itext] = np.nan
            left_index_finger_distances[ilayout, itext] = np.nan
            right_index_finger_distances[ilayout, itext] = np.nan
            left_little_finger_taps[ilayout, itext] = np.nan
            right_little_finger_taps[ilayout, itext] = np.nan
            same_finger_different_key_taps[ilayout, itext] = np.nan
            hand_balances[ilayout, itext] = np.nan

# Store KLA metrics
kla_score_matrices = [
    total_finger_distances,
    left_little_finger_distances,
    right_little_finger_distances,
    left_index_finger_distances,
    right_index_finger_distances,
    left_little_finger_taps,
    right_little_finger_taps,
    same_finger_different_key_taps,
    hand_balances
]

kla_matrix_labels = [
    "Total finger distances",
    "Left little finger distances", 
    "Right little finger distances",
    "Left index finger distances",
    "Right index finger distances",
    "Left little finger taps",
    "Right little finger taps",
    "Same finger different key taps",
    "Hand balances"
]

kla_file_names = [
    "kla_total_finger_distances",
    "kla_left_little_finger_distances",
    "kla_right_little_finger_distances", 
    "kla_left_index_finger_distances",
    "kla_right_index_finger_distances",
    "kla_left_little_finger_taps",
    "kla_right_little_finger_taps",
    "kla_same_finger_different_key_taps",
    "kla_hand_balances"
]

kla_divide_bys = [100, 100, 100, 100, 100, 1, 1, 1, 1]
kla_display_ints = [True, True, True, True, True, True, True, True, False]

print("\nGenerating KLA CSV tables...")

# Generate CSV files for each KLA metric
for imatrix, (kla_score_matrix, matrix_label, file_name) in enumerate(zip(kla_score_matrices, kla_matrix_labels, kla_file_names)):
    
    # Create DataFrame with layouts as rows and texts as columns
    kla_df = pd.DataFrame(
        kla_score_matrix, 
        index=layout_abbrs_KLA,
        columns=text_abbrs
    )
    kla_df.index.name = 'Layout'
    
    # Save to CSV
    csv_filename = f"output/csv/{file_name}.csv"
    kla_df.to_csv(csv_filename)
    print(f"Saved: {csv_filename}")
    save_dataframe_as_markdown(kla_df, csv_filename, matrix_label, 
                             divide_by=kla_divide_bys[imatrix], 
                             display_int=kla_display_ints[imatrix])
    save_dataframe_as_heatmap(kla_df, csv_filename, matrix_label, divide_by=kla_divide_bys[imatrix])

print(f"\nGenerated KLA CSV tables:")
print(f"- output/kla_*.csv (9 tables)")
print(f"\nAll tables have layouts as rows and text sources as columns.")

# Optional: Print summary statistics
print(f"\nKLA Summary Statistics:")
print(f"Total layouts processed: {len(layout_abbrs_KLA)}")
print(f"Total text sources: {len(text_abbrs)}")
print(f"KLA metrics calculated: {len(kla_matrix_labels)}")

print(f"\nGenerated files:")
print(f"- output/csv/score_layout_results.csv (1 file)")
print(f"- output/csv/dvorak9_*.csv (10 files)")  
print(f"- output/csv/kla_*.csv (9 files)")
print(f"- output/md/*.md (20 markdown files)")
print(f"- output/png/*.png (20 heatmap files)")
print(f"- output/bigrams_text_data.txt (generated bigram text)")
print(f"\nAll tables have layouts as rows and text sources as columns (including Bigrams).")