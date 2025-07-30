# Compare keyboard layouts
https://github.com/binarybottle/compare_keyboard_layouts

(c) 2021-2025 Arno Klein (arnoklein.info), MIT license

----

# Contents
1. [Layouts](#layouts")
2. [Text data](#text-data")
3. [Score layouts using Dvorak-11 scoring model](#dvorak-11")
    - [Dvorak-11 scores](#dvorak11-scores)
4. [Score layouts using Keyboard Layout Analyzer (KLA) measures](#kla-scores)
    - [KLA distance and tapping measures](#kla-tapping)
5. [Miscellaneous tests](#misc) 
    - [Inward roll frequencies](#rolls)
    - [Interkey speed estimates](#speed)
    - [Carpalx scores](#carpalx)

--------------------------------------------------------------------------------

## Why compare keyboard layouts? <a name="why">

Below we compare different prominent keyboard layouts (Colemak, Dvorak, QWERTY, etc.) 
using some large, representative, publicly available data 
(all text sources are listed below and available on 
[GitHub](https://github.com/binarybottle/text_data)).

---

## Keyboard layouts <a name="layouts">

| Layout | Year | Website |
| --- | --- | --- |
| Engram | 2021 | https://engram.dev |
| [Halmak 2.2](https://keyboard-design.com/letterlayout.html?layout=halmak-2-2.en.ansi) | 2016 | https://github.com/MadRabbit/halmak |
| [Hieamtsrn](https://www.keyboard-design.com/letterlayout.html?layout=hieamtsrn.en.ansi) | 2014 | https://mathematicalmulticore.wordpress.com/the-keyboard-layout-project/#comment-4976 |
| [Colemak Mod-DH](https://keyboard-design.com/letterlayout.html?layout=colemak-mod-DH-full.en.ansi) | 2014 | https://colemakmods.github.io/mod-dh/ | 
| [Norman](https://keyboard-design.com/letterlayout.html?layout=norman.en.ansi) | 2013 | https://normanlayout.info/ |
| [Workman](https://keyboard-design.com/letterlayout.html?layout=workman.en.ansi) | 2010 | https://workmanlayout.org/ | 
| [MTGAP 2.0](https://www.keyboard-design.com/letterlayout.html?layout=mtgap-2-0.en.ansi) | 2010 | https://mathematicalmulticore.wordpress.com/2010/06/21/mtgaps-keyboard-layout-2-0/ |
| [QGMLWB](https://keyboard-design.com/letterlayout.html?layout=qgmlwb.en.ansi) | 2009 | http://mkweb.bcgsc.ca/carpalx/?full_optimization | 
| [Colemak](https://keyboard-design.com/letterlayout.html?layout=colemak.en.ansi) | 2006 | https://colemak.com/ | 
| [Asset](https://keyboard-design.com/letterlayout.html?layout=asset.en.ansi) | 2006 | http://millikeys.sourceforge.net/asset/ | 
| Capewell-Dvorak | 2004 | http://michaelcapewell.com/projects/keyboard/layout_capewell-dvorak.htm |
| [Klausler](https://www.keyboard-design.com/letterlayout.html?layout=klausler.en.ansi) | 2002 | https://web.archive.org/web/20031001163722/http://klausler.com/evolved.html |
| [Dvorak](https://keyboard-design.com/letterlayout.html?layout=dvorak.en.ansi) | 1936 | https://en.wikipedia.org/wiki/Dvorak_keyboard_layout | 
| [QWERTY](https://keyboard-design.com/letterlayout.html?layout=qwerty.en.ansi) | 1873 | https://en.wikipedia.org/wiki/QWERTY |

---

## Scoring methods <a name="scoring">

| Layout | Year | Website |
| --- | --- | --- |
| Engram | 2021 | https://engram.dev |
| [Halmak 2.2](https://keyboard-design.com/letterlayout.html?layout=halmak-2-2.en.ansi) | 2016 | https://github.com/MadRabbit/halmak |
| [Hieamtsrn](https://www.keyboard-design.com/letterlayout.html?layout=hieamtsrn.en.ansi) | 2014 | https://mathematicalmulticore.wordpress.com/the-keyboard-layout-project/#comment-4976 |
| [Colemak Mod-DH](https://keyboard-design.com/letterlayout.html?layout=colemak-mod-DH-full.en.ansi) | 2014 | https://colemakmods.github.io/mod-dh/ | 
| [Norman](https://keyboard-design.com/letterlayout.html?layout=norman.en.ansi) | 2013 | https://normanlayout.info/ |
| [Workman](https://keyboard-design.com/letterlayout.html?layout=workman.en.ansi) | 2010 | https://workmanlayout.org/ | 
| [MTGAP 2.0](https://www.keyboard-design.com/letterlayout.html?layout=mtgap-2-0.en.ansi) | 2010 | https://mathematicalmulticore.wordpress.com/2010/06/21/mtgaps-keyboard-layout-2-0/ |
| [QGMLWB](https://keyboard-design.com/letterlayout.html?layout=qgmlwb.en.ansi) | 2009 | http://mkweb.bcgsc.ca/carpalx/?full_optimization | 
| [Colemak](https://keyboard-design.com/letterlayout.html?layout=colemak.en.ansi) | 2006 | https://colemak.com/ | 
| [Asset](https://keyboard-design.com/letterlayout.html?layout=asset.en.ansi) | 2006 | http://millikeys.sourceforge.net/asset/ | 
| Capewell-Dvorak | 2004 | http://michaelcapewell.com/projects/keyboard/layout_capewell-dvorak.htm |
| [Klausler](https://www.keyboard-design.com/letterlayout.html?layout=klausler.en.ansi) | 2002 | https://web.archive.org/web/20031001163722/http://klausler.com/evolved.html |
| [Dvorak](https://keyboard-design.com/letterlayout.html?layout=dvorak.en.ansi) | 1936 | https://en.wikipedia.org/wiki/Dvorak_keyboard_layout | 
| [QWERTY](https://keyboard-design.com/letterlayout.html?layout=qwerty.en.ansi) | 1873 | https://en.wikipedia.org/wiki/QWERTY |

---

## Text data for scoring <a name="text-data">

N-gram letter frequencies<br>
    - [Peter Norvig's analysis](http://www.norvig.com/mayzner.html) of data from Google's book scanning project

| Text source | Information |
| --- | --- |
| "Alice in Wonderland" | Alice in Wonderland (Ch.1) |
| "Memento screenplay" | [Memento screenplay](https://www.dailyscript.com/scripts/memento.html) |
| "100K tweets" | 100,000 tweets from: [Sentiment140 dataset](https://data.world/data-society/twitter-user-data) training data |
| "20K tweets" | 20,000 tweets from [Gender Classifier Data](https://www.kaggle.com/crowdflower/twitter-user-gender-classification) |
| "MASC tweets" | [MASC](http://www.anc.org/data/masc/corpus/) tweets (cleaned of html markup) |
| "MASC spoken" | [MASC](http://www.anc.org/data/masc/corpus/) spoken transcripts (phone and face-to-face: 25,783 words) |
| "COCA blogs" | [Corpus of Contemporary American English](https://www.english-corpora.org/coca/) [blog samples](https://www.corpusdata.org/) |
| "Rosetta" | "Tower of Hanoi" (programming languages A-Z from [Rosetta Code](https://rosettacode.org/wiki/Towers_of_Hanoi)) |
| "Monkey text" | Ian Douglas's English-generated [monkey0-7.txt corpus](https://zenodo.org/record/4642460) |
| "Coder text" | Ian Douglas's software-generated [coder0-7.txt corpus](https://zenodo.org/record/4642460) |
| "iweb cleaned corpus" | First 150,000 lines of Shai Coleman's [iweb-corpus-samples-cleaned.txt](https://colemak.com/pub/corpus/iweb-corpus-samples-cleaned.txt.xz) |

Reference for Monkey and Coder texts:
Douglas, Ian. (2021, March 28). Keyboard Layout Analysis: Creating the Corpus, Bigram Chains, and Shakespeare's Monkeys (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.4642460

---

## Results <a name="results">

### Engram scores <a name="engram-scores">
 
---

### Dvorak-9 scores <a name="dvorak9-scores">
 
---

### Keyboard Layout Analyzer scores <a name="kla-scores"> 

[Keyboard Layout Analyzer](http://patorjk.com/keyboard-layout-analyzer/) (KLA) scores for the same text sources
    - Finger distances (cm)
    - Number of taps
    - Number of same-finger taps 

The optimal layout score is based on a weighted calculation that factors in the distance your fingers moved (33%), how often you use particular fingers (33%), and how often you switch fingers and hands while typing (34%).
    
Engram scores highest for 7 of the 9 and second highest for 2 of the 9 text sources; Engram scores third and fourth highest for the two software sources, "Coder" and "Rosetta" (higher scores are better):

| Layout | Alice in Wonderland | Memento screenplay | 100K tweets | 20K tweets | MASC tweets | MASC spoken | COCA blogs | iweb | Monkey | Coder | Rosetta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engram    | 70.13 | 57.16 | 64.64 | 58.58 | 60.24 | 64.39 | 69.66 | 68.25 | 67.66 | 46.81 | 47.69 |
| Halmak    | 66.25 | 55.03 | 60.86 | 55.53 | 57.13 | 62.32 | 67.29 | 65.50 | 64.75 | 45.68 | 47.60 |
| Hieamtsrn | 69.43 | 56.75 | 64.40 | 58.95 | 60.47 | 64.33 | 69.93 | 69.15 | 68.30 | 46.01 | 46.48 | 
| Colemak Mod-DH | 65.74 | 54.91 | 60.75 | 54.94 | 57.15 | 61.29 | 67.12 | 65.98 | 64.85 | 47.35 | 48.50 |
| Norman    | 62.76 | 52.33 | 57.43 | 53.24 | 53.90 | 59.97 | 62.80 | 60.90 | 59.82 | 43.76 | 46.01 |
| Workman   | 64.78 | 54.29 | 59.98 | 55.81 | 56.25 | 61.34 | 65.27 | 63.76 | 62.90 | 45.33 | 47.76 |
| MTGAP 2.0 | 66.13 | 53.78 | 59.87 | 55.30 | 55.81 | 60.32 | 65.68 | 63.81 | 62.74 | 45.38 | 44.34 | 
| QGMLWB    | 65.45 | 54.07 | 60.51 | 56.05 | 56.90 | 62.23 | 66.26 | 64.76 | 63.91 | 46.38 | 45.72 |
| Colemak   | 65.83 | 54.94 | 60.67 | 54.97 | 57.04 | 61.36 | 67.14 | 66.01 | 64.91 | 47.30 | 48.65 |
| Asset     | 64.60 | 53.84 | 58.66 | 54.72 | 55.35 | 60.81 | 64.71 | 63.17 | 62.44 | 45.54 | 47.52 |
| Capewell-Dvorak | 66.94 | 55.66 | 62.14 | 56.85 | 57.99 | 62.83 | 66.95 | 65.23 | 64.70 | 45.30 | 45.62 |
| Klausler  | 68.24 | 59.91 | 62.57 | 56.45 | 58.34 | 64.04 | 68.34 | 66.89 | 66.31 | 46.83 | 45.66 |
| Dvorak    | 65.86 | 58.18 | 60.93 | 55.56 | 56.59 | 62.75 | 66.64 | 64.87 | 64.26 | 45.46 | 45.55 | 
| QWERTY    | 53.06 | 43.74 | 48.28 | 44.99 | 44.59 | 51.79 | 52.31 | 50.19 | 49.18 | 38.46 | 39.89 | 

---

### Carpalx scores <a name="carpalx-scores">

[Carpalx](http://mkweb.bcgsc.ca/carpalx/?keyboard_layouts) scores
are computed based on literature from the Gutenberg Project.
  
| Layout | home row use (%) | hand symmetry (%, right<0) | hand switching (%) | finger switching (%) | hand runs without row jumps (%) | base effort  | penalties | path effort | total effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engram | 58 | -99 | 61 | 93 | 82 | 0.419 | 0.950 | 0.803 | 2.171 |
| Hieamtsrn | 65 | -96 | 59 | 93 | 85 | 0.38 | 0.848 | 0.720 | 1.948 |
| Halmak | 64 | 99 | 63 | 93 | 81 | 0.325 | 1.175 | 0.823 | 2.322 |
| Norman | 68 | 95 | 52 | 90 | 77 | 0.342 | 0.812 | 0.838 | 1.992 |
| Workman | 68 | 95 | 52 | 93 | 79 | 0.336 | 0.848 | 0.809 | 1.993 |
| MTGAP 2.0 | 68 | 98 | 48 | 93 | 76 | 0.327 | 0.839 | 0.815 | 1.981 |
| QGMLWB | 74 | -97 | 57 | 91 | 84 | 0.382 | 0.570 | 0.716 | 1.668 |
| BEAKL 15 | 57 | -95 | 59 | 91 | 80 | 0.473 | 0.663 | 0.809 | 1.945 |
| Colemak Mod-DH | 68 | -94 | 52 | 93 | 78 | 0.335 | 0.842 | 0.781 | 1.958 |
| Colemak | 74 | -94 | 52 | 93 | 83 | 0.344 | 0.763 | 0.735 | 1.842 |
| Asset | 74 | 96 | 52 | 91 | 82 | 0.356 | 0.766 | 0.772 | 1.894 |
| Capewell-Dvorak | 71 | -91 | 59 | 92 | 82 | 0.333 | 0.878 | 0.774 | 1.985 |
| Klausler | 74 | -94 | 62 | 93 | 86 | 0.341 | 0.797 | 0.729 | 1.867 |
| Dvorak | 71 | -86 | 62 | 93 | 84 | 0.397 | 0.937 | 0.765 | 2.098 |
| QWERTY | 34 | 85 | 51 | 89 | 68 | 1 | 1 | 1 | 3 |

---
