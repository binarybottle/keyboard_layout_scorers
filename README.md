# Compare keyboard layouts
https://github.com/binarybottle/compare_keyboard_layouts

(c) 2021-2025 Arno Klein (arnoklein.info), MIT license

--------------------------------------------------------------------------------

# Contents
1. [Why compare keyboard layouts?](#why)
2. [Why compare keyboard layouts?](#why)
3. [Summary of steps and results](#summary)

## Why compare keyboard layouts? <a name="why">
## Keyboard layout scoring methods <a name="scoring-methods">
## Text data used to calculate scores <a name="text-data">
## Results <a name="results">
### Dvorak-inspired scores <a name="dvorak-scores">
### Keyboard Layout Analyzer scores <a name="kla-scores">
### Keyboard Layout Analyzer consecutive same-finger key presses <a name="kla-same-finger">
### Inward roll frequencies <a name="inward-rolls">

---

## Why compare keyboard layouts? <a name="why">

Below we compare different prominent key layouts (Colemak, Dvorak, QWERTY, etc.) 
using some large, representative, publicly available data 
(all text sources are listed below and available on 
[GitHub](https://github.com/binarybottle/text_data)).

---

## Keyboard layout scoring methods <a name="scoring-methods">

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

## Text data used to calculate scores <a name="text-data">

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

### Dvorak-inspired scores <a name="dvorak-scores">
 
Dvorak-inspired scoring method scores:

Guiding criteria:

    1.  Assign letters to keys that don't require lateral finger movements.
    2.  Promote alternating between hands over uncomfortable same-hand transitions.
    3.  Assign the most common letters to the most comfortable keys.
    4.  Arrange letters so that more frequent bigrams are easier to type.
    5.  Promote little-to-index-finger roll-ins over index-to-little-finger roll-outs.
    6.  Balance finger loads according to their relative strength.
    7.  Avoid stretching shorter fingers up and longer fingers down.
    8.  Avoid using the same finger.
    9.  Avoid skipping over the home row.
    10. Assign the most common punctuation to keys in the middle of the keyboard.
    11. Assign easy-to-remember symbols to the Shift-number keys.
    
| Layout | Google bigrams | Alice | Memento | Tweets_100K | Tweets_20K | Tweets_MASC | Spoken_MASC | COCA_blogs | iweb | Monkey | Coder | Rosetta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engram | 62.48 | 61.67 | 62.30 | 63.03 | 60.28 | 62.49 | 61.56 | 62.19 | 62.38 | 62.23 | 62.51 | 62.48 |
| Halmak | 62.40 | 61.60 | 62.23 | 62.93 | 60.26 | 62.43 | 61.51 | 62.13 | 62.31 | 62.16 | 62.46 | 62.40 |
| Hieamtsrn | 62.39 | 61.64 | 62.27 | 62.99 | 60.27 | 62.47 | 61.53 | 62.16 | 62.35 | 62.20 | 62.49 | 62.39 |
| Norman | 62.35 | 61.57 | 62.20 | 62.86 | 60.21 | 62.39 | 61.47 | 62.08 | 62.27 | 62.12 | 62.40 | 62.35 |
| Workman | 62.37 | 61.59 | 62.22 | 62.91 | 60.23 | 62.41 | 61.49 | 62.10 | 62.29 | 62.14 | 62.43 | 62.37 |
| MTGap 2.0 | 62.32 | 61.59 | 62.21 | 62.88 | 60.22 | 62.39 | 61.49 | 62.09 | 62.28 | 62.13 | 62.42 | 62.32 |
| QGMLWB | 62.31 | 61.58 | 62.21 | 62.90 | 60.25 | 62.40 | 61.49 | 62.10 | 62.29 | 62.14 | 62.43 | 62.31 |
| Colemak Mod-DH | 62.36 | 61.60 | 62.22 | 62.90 | 60.26 | 62.41 | 61.49 | 62.12 | 62.30 | 62.16 | 62.44 | 62.36 |
| Colemak | 62.36 | 61.58 | 62.20 | 62.89 | 60.25 | 62.40 | 61.48 | 62.10 | 62.29 | 62.14 | 62.43 | 62.36 |
| Asset | 62.34 | 61.56 | 62.18 | 62.86 | 60.25 | 62.37 | 61.46 | 62.07 | 62.25 | 62.10 | 62.39 | 62.34 |
| Capewell-Dvorak | 62.29 | 61.56 | 62.17 | 62.86 | 60.20 | 62.36 | 61.47 | 62.06 | 62.24 | 62.10 | 62.37 | 62.29 |
| Klausler | 62.34 | 61.58 | 62.20 | 62.89 | 60.25 | 62.39 | 61.48 | 62.09 | 62.27 | 62.12 | 62.41 | 62.34 |
| Dvorak | 62.31 | 61.56 | 62.17 | 62.85 | 60.23 | 62.35 | 61.46 | 62.06 | 62.24 | 62.09 | 62.35 | 62.31 |
| QWERTY | 62.19 | 61.49 | 62.08 | 62.72 | 60.17 | 62.25 | 61.39 | 61.96 | 62.13 | 61.99 | 62.25 | 62.19 |

---

### Keyboard Layout Analyzer scores <a name="kla-scores"> 

[Keyboard Layout Analyzer](http://patorjk.com/keyboard-layout-analyzer/) (KLA) scores for the same text sources
    
> The optimal layout score is based on a weighted calculation that factors in the distance your fingers moved (33%), how often you use particular fingers (33%), and how often you switch fingers and hands while typing (34%).
    
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

### Keyboard Layout Analyzer consecutive same-finger key presses <a name="kla-same-finger">

KLA (and other) distance measures may not accurately reflect natural typing, 
so below is a more reliable measure of one source of effort and strain -- 
the tally of consecutive key presses with the same finger for different keys. 
Engram scores lowest for 6 of the 11 texts, second lowest for two texts, 
and third or fifth lowest for three texts, two of which are software text 
sources (lower scores are better):

| Layout | Alice | Memento | Tweets_100K | Tweets_20K | Tweets_MASC | Spoken_MASC | COCA_blogs | iweb | Monkey | Coder | Rosetta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engram | 216 | 11476 | 320406 | 120286 | 7728 | 3514 | 137290 | 1064640 | 37534 | 125798 | 5822 |
| Halmak | 498 | 13640 | 484702 | 170064 | 11456 | 5742 | 268246 | 2029634 | 68858 | 144790 | 5392 |
| Hieamtsrn | 244 | 12096 | 311000 | 119490 | 8316 | 3192 | 155674 | 1100116 | 40882 | 158698 | 7324 |
| Norman | 938 | 20012 | 721602 | 213890 | 16014 | 9022 | 595168 | 3885282 | 135844 | 179752 | 7402 |
| Workman | 550 | 13086 | 451280 | 136692 | 10698 | 6156 | 287622 | 1975564 | 71150 | 132526 | 5550 |
| MTGap 2.0 | 226 | 14550 | 397690 | 139130 | 10386 | 6252 | 176724 | 1532844 | 58144 | 138484 | 7272 |
| QGMLWB | 812 | 17820 | 637788 | 189700 | 14364 | 7838 | 456442 | 3027530 | 100750 | 149366 | 8062 |
| Colemak Mod-DH | 362 | 10960 | 352578 | 151736 | 9298 | 4644 | 153984 | 1233770 | 47438 | 117842 | 5328 |
| Colemak | 362 | 10960 | 352578 | 151736 | 9298 | 4644 | 153984 | 1233770 | 47438 | 117842 | 5328 |
| Asset | 520 | 12519 | 519018 | 155246 | 11802 | 5664 | 332860 | 2269342 | 77406 | 140886 | 6020 |
| Capewell-Dvorak | 556 | 14226 | 501178 | 163878 | 12214 | 6816 | 335056 | 2391416 | 78152 | 151194 | 9008 |
| Klausler | 408 | 14734 | 455658 | 174998 | 11410 | 5212 | 257878 | 1794604 | 59566 | 135782 | 7444 |
| Dvorak | 516 | 13970 | 492604 | 171488 | 12208 | 5912 | 263018 | 1993346 | 64994 | 142084 | 6484 |

---
  
### Inward roll frequencies <a name="inward-rolls">

Here we tally the number of bigrams (in billions of instances from Norvig's 
analysis of Google data) that engage inward rolls (little-to-index sequences), 
within the four columns of one hand, or any column across two hands. 
Engram scores second highest for 32 keys and highest for 24 keys, 
where the latter ensures that we are comparing Engram's letters with letters 
in other layouts (higher scores are better):
    
Total inward roll frequency in billions

    Layout             32 / 24 keys
    Engram:          4.64 / 4.51
    Halmak:          4.59 / 4.25
    Hieamtsrn:       4.69 / 4.16
    Norman:          3.99 / 3.61
    Workman:         4.16 / 3.63
    MTGap 2.0:       3.96 / 3.58
    QGMLWB:          4.36 / 2.81
    Colemak Mod-DH:  4.15 / 3.51
    Colemak:         4.17 / 3.16
    Asset:           4.03 / 3.05
    Capewell-Dvorak: 4.39 / 3.66
    Klausler:        4.42 / 3.52
    Dvorak:          4.40 / 3.20
    QWERTY:          3.62 / 2.13
   
---