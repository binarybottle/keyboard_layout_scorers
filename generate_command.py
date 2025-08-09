#!/usr/bin/env python3
"""
Generate commands for keyboard layout comparison workflow.

This script creates ready-to-run commands for comparing standard layouts
using the two-step workflow: score_layouts.py → compare_layouts.py

Usage:
    python3 generate_command.py [--output-dir OUTPUT_DIR] [--scores-file SCORES_FILE]
    python3 generate_command.py --python-cmd "poetry run python3"  # if using poetry

1. Generate the workflow commands
>> python generate_command.py

2. Run the complete workflow
>> ./output/compare_standard_layouts.sh

# Or run steps individually:
1. Generate scores
>> python score_layouts.py --compare [layouts...] --csv-output > layout_scores.csv

2. Visualize (multiple options)
>> python compare_layouts.py --tables layout_scores.csv --output comparison
>> python compare_layouts.py --tables layout_scores.csv --use-raw --output comparison_raw
>> python compare_layouts.py --tables layout_scores.csv --verbose

"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate layout comparison workflow commands')
    parser.add_argument('--output-dir', default='output', 
                       help='Directory for output command files')
    parser.add_argument('--scores-file', default='layout_scores.csv',
                       help='Name for the scores CSV file (default: layout_scores.csv)')
    parser.add_argument('--python-cmd', default='python3',
                       help='Python command to use (default: python3, try "poetry run python3" if using poetry)')
    
    args = parser.parse_args()
    
    # Standard keyboard layouts with their layout strings
    standard_layouts = {
        'Halmak-2.2': "wlrbz;qudjshnt,.aeoifmvc/gpxky['",
        'Hieamtsrn': "byou'kdclphiea,mtsrnx-\".?wgfjzqv",
        'Colemak-DH': "qwfpbjluy;arstgmneiozxcdvkh,./['",
        'Norman': "qwdfkjurl;asetgyniohzxcvbpm,./['",
        'Workman': "qdrwbjfup;ashtgyneoizxmcvkl,./['",
        'MTGAP-2.0': ",fhdkjcul.oantgmseriqxbpzyw'v;['",
        'QGMLWB': "qgmlwbyuv;dstnriaeohzxcfjkp,./['",
        'Colemak': "qwfpgjluy;arstdhneiozxcvbkm,./['",
        'Asset': "qwfgjypul;asetdhniorzxcvbkm,./['",
        'Capewell-Dvorak': "',.pyqfgrkoaeiudhtnszxcvjlmwb;['",
        'Klausler': "k,uypwlmfcoaeidrnthsq.';zxvgbj['",
        'Dvorak': "',.pyfgcrlaoeuidhtns;qjkxbmwvz['",
        'QWERTY': "qwertyuiopasdfghjkl;zxcvbnm,./['"
    }
    
    # Create layout specifications for the scoring command
    layout_specs = []
    for name, layout_string in standard_layouts.items():
        # Escape double quotes in the layout string if any
        escaped_layout = layout_string.replace('"', '\\"')
        layout_specs.append(f'{name}:"{escaped_layout}"')
    
    # Build the two-step workflow commands
    
    # Step 1: Generate scores
    step1_parts = [
        f"{args.python_cmd} score_layouts.py",
        "--compare",
        " ".join(layout_specs),
        "--csv-output"
    ]
    step1_command = " ".join(step1_parts) + f" > {args.scores_file}"
    
    # Step 2: Generate visualizations  
    step2_command = f"{args.python_cmd} compare_layouts.py --tables {args.scores_file} --output layout_comparison"
    
    # Alternative step 2 commands
    step2_raw = f"{args.python_cmd} compare_layouts.py --tables {args.scores_file} --use-raw --output layout_comparison_raw"
    step2_verbose = f"{args.python_cmd} compare_layouts.py --tables {args.scores_file} --verbose"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Write to shell script file
    script_file = output_dir / 'compare_standard_layouts.sh'
    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Standard Keyboard Layout Comparison Workflow\n")
        f.write("# Generated two-step workflow for comparing standard keyboard layouts\n")
        f.write(f"# Python command used: {args.python_cmd}\n")
        f.write("# If you get 'command not found', try regenerating with --python-cmd option\n")
        f.write(f"# Number of layouts: {len(standard_layouts)}\n")
        f.write("# Layouts included:\n")
        for i, (name, layout_string) in enumerate(standard_layouts.items(), 1):
            f.write(f"#   {i:2d}. {name.replace('_', ' ')}: {layout_string}\n")
        f.write("\n")
        f.write("echo 'Step 1: Generating layout scores...'\n")
        f.write(step1_command + "\n")
        f.write("echo 'Scores saved to " + args.scores_file + "'\n")
        f.write("\n")
        f.write("echo 'Step 2: Creating visualizations...'\n")
        f.write(step2_command + "\n")
        f.write("echo 'Visualizations saved as layout_comparison_parallel.png and layout_comparison_heatmap.png'\n")
        f.write("\n")
        f.write("echo 'Workflow complete!'\n")
    
    # Write to text file with individual commands
    commands_file = output_dir / 'layout_comparison_commands.txt'
    with open(commands_file, 'w') as f:
        f.write("# Standard Keyboard Layout Comparison Commands\n")
        f.write("# Two-step workflow: score_layouts.py → compare_layouts.py\n")
        f.write(f"# Python command used: {args.python_cmd}\n")
        f.write("# If you get 'command not found', try regenerating with --python-cmd option\n")
        f.write(f"# Number of layouts: {len(standard_layouts)}\n")
        f.write("# Layouts included:\n")
        for i, (name, layout_string) in enumerate(standard_layouts.items(), 1):
            f.write(f"#   {i:2d}. {name.replace('_', ' ')}: {layout_string}\n")
        f.write("\n")
        f.write("# STEP 1: Generate layout scores\n")
        f.write("# This creates a CSV file with layout_name,scorer,weighted_score,raw_score\n")
        f.write(step1_command + "\n")
        f.write("\n")
        f.write("# STEP 2: Create visualizations\n")
        f.write("# Basic visualization (weighted scores)\n")
        f.write(step2_command + "\n")
        f.write("\n")
        f.write("# Alternative visualization options:\n")
        f.write("# Using raw scores instead of weighted scores\n")
        f.write(step2_raw + "\n")
        f.write("\n")
        f.write("# Verbose output with detailed statistics\n")
        f.write(step2_verbose + "\n")
        f.write("\n")
        f.write("# View just the scores without visualization\n")
        f.write(f"cat {args.scores_file}\n")
    
    # Make shell script executable
    script_file.chmod(0o755)
    
    print(f"Layout comparison workflow files generated:")
    print(f"  Shell script: {script_file}")
    print(f"  Commands file: {commands_file}")
    print(f"  Scores file will be: {args.scores_file}")
    print()
    print(f"Workflow includes {len(standard_layouts)} standard layouts:")
    
    # Print layout list
    for i, (name, layout_string) in enumerate(standard_layouts.items(), 1):
        display_name = name.replace('_', ' ')
        print(f"  {i:2d}. {display_name}")
    
    print()
    print("Usage:")
    print(f"  1. Run the shell script: ./{script_file}")
    print(f"  2. Or run commands individually from: {commands_file}")
    print()
    print("If using poetry, regenerate with:")
    print('  python3 generate_command.py --python-cmd "poetry run python3"')
    print()
    print("Output files will be:")
    print(f"  - {args.scores_file} (CSV with all scores)")
    print("  - layout_comparison_parallel.png (parallel coordinates plot)")
    print("  - layout_comparison_heatmap.png (heatmap visualization)")


if __name__ == '__main__':
    main()