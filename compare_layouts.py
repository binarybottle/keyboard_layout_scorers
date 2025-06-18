"""
Keyboard layout comparison

See README for information.

(c) 2021-2025 Arno Klein (arnoklein.info), MIT license

Each layout is represented by a list of 32 characters in the following order:

    1  2  3  4   25  28   13 14 15 16   31
    5  6  7  8   26  29   17 18 19 20   32
    9 10 11 12   27  30   21 22 23 24 

"""
import os
import subprocess
import pandas as pd
import requests
from io import StringIO
import tempfile
import re

# DEBUG: Set to a number to limit text length for faster testing, or None for full text
DEBUG_TEXT_LIMIT = 100  # Change to None for full text processing
DEBUG_TIMEOUT = 60  # Increase timeout for debugging

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

# Select text samples:
text_indices = range(len(text_names))
text_names_copy = []
text_abbrs_copy = []
for text_index in text_indices:
    text_names_copy.append(text_names[text_index])
    text_abbrs_copy.append(text_abbrs[text_index])
text_names = text_names_copy
text_abbrs = text_abbrs_copy    

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
            'empirical_score',           # Total weighted score (first value)
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
# Score layouts
#===============================================================================
print("Loading text files from local directory...")
texts = load_local_texts(text_abbrs)

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
score_layout_df.to_csv('score_layout_results.csv')
print("Saved: score_layout_results.csv")

# Create dvorak9_scorer.py results tables
dvorak9_metrics = [
    'empirical_score',  # Total score
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
    filename = f"dvorak9_{metric.replace('individual_', '')}.csv"
    metric_df.to_csv(filename)
    print(f"Saved: {filename}")

print(f"\nGenerated CSV tables:")
print(f"- score_layout_results.csv (1 table)")
print(f"- dvorak9_*.csv (10 tables)")
print(f"\nTables have layouts as rows and text sources as columns.")