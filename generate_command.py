#!/usr/bin/env python3
"""
Generate layout_scorer.py command for standard keyboard layouts.

This script creates a ready-to-run command file for comparing standard layouts
using the layout_scorer.py tool.

Usage:
    python3 generate_standard_layouts_command.py [--output-dir OUTPUT_DIR]
"""

import argparse
from pathlib import Path

text_file = "/Users/arno/Software/text_data/samples/COCA_corpusdata.org_sample_text_blog_cleaned.txt"

def main():
    parser = argparse.ArgumentParser(description='Generate layout_scorer.py command for standard layouts')
    parser.add_argument('--output-dir', default='output', 
                       help='Directory for output command file')
    
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
    
    # Create layout specifications for the command
    layout_specs = []
    for name, layout_string in standard_layouts.items():
        # Escape double quotes in the layout string if any
        escaped_layout = layout_string.replace('"', '\\"')
        layout_specs.append(f'{name}:"{escaped_layout}"')
    
    # Build the complete command
    command_parts = [
        "python layout_scorer.py",
        "--compare",
        " ".join(layout_specs),
        "--csv results.csv",
        '--text-file', text_file
    ]
    
    command = " ".join(command_parts)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Write to file
    command_file = output_dir / 'standard_layouts_scorer_command.txt'
    with open(command_file, 'w') as f:
        f.write("# Standard Layouts Scorer Command\n")
        f.write("# Generated command for comparing standard keyboard layouts\n")
        f.write(f"# Number of layouts: {len(standard_layouts)}\n")
        f.write("# Layouts included:\n")
        for i, (name, layout_string) in enumerate(standard_layouts.items(), 1):
            f.write(f"#   {i:2d}. {name.replace('_', ' ')}: {layout_string}\n")
        f.write("# Usage: Copy and run this command in your layout_scorer directory\n\n")
        f.write(command)
        f.write("\n")
    
    print(f"Standard layouts scorer command saved to: {command_file}")
    print(f"Command includes {len(standard_layouts)} standard layouts:")
    
    # Print layout list
    for i, (name, layout_string) in enumerate(standard_layouts.items(), 1):
        display_name = name.replace('_', ' ')
        print(f"  {i:2d}. {display_name}")
    
    print(f"\nGenerated command file: {command_file}")
    print("Copy the command from the file and run it in your layout_scorer directory.")


if __name__ == '__main__':
    main()