# 15-110 Final Project: Language Model

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Data Sources](#data-sources)
5. [How It Works](#how-it-works)
6. [Example Usage](#example-usage)
7. [Example Output](#example-output)
8. [Custom Text Analysis](#custom-text-analysis)

## Overview
This project implements a language model in Python that analyzes two corpora of text. It generates text in the style of the corpus authors and compares the probabilities and frequencies of unigrams and bigrams within the corpora.

## Features
- **Text Analysis**: Analyze and compare the linguistic patterns of two corpora.
- **Text Generation**: Generate text that mimics the style of the authors in the corpora.
- **Statistical Comparison**: Compute and compare unigram and bigram probabilities and frequencies.

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/cdundon123/langmodel_110.git
    ```

2. **Install Dependencies**: Ensure you have Python 3.8 installed.
    The project uses `matplotlib` for generating visualizations. Ensure it is installed to view the outputs:
    ```bash
    pip install matplotlib
    ```
3. **Run the Language Model**:  Execute the main script to analyze the corpora or generate text:
    ```bash
    python language_model/language_func.py.py
    ```
4. **Run the Tests**: Verify the functionality of the language model using:
    ```bash
    python -m unittest language_model/tests.py
    ```

## Data Sources
The data/ directory contains the following text corpora:
- **Andersen's Fairy Tales:** Cleaned and raw versions of Hans Christian Andersen's works.
- **Grimm's Fairy Tales:** Cleaned and raw versions of the Brothers Grimm's works.
- **Hamilton:** A text corpus for additional analysis
- **Test Files:** Small sample files (```test1.txt```, ```test2.txt```) for debugging and testing.

## How It Works
1. __Text Preprocessing:__ The raw text is cleaned and tokenized into unigrams and bigrams.
2. __Probability Calculation:__ The model calculates unigram and bigram probabilities for each corpus.
3. __Text Generation:__ Using the probabilities, the model generates text in the style of the corpus authors.

## Example Usage
To generate text in the style of a specific corpus:
```bash
python [language_func.py](http://_vscodecontentref_/6) --generate --corpus [andersen_clean.txt](http://_vscodecontentref_/7)
```

To compare unigram and bigram probabilities:
```bash
python [language_func.py](http://_vscodecontentref_/8) --compare --corpus1 [andersen_clean.txt](http://_vscodecontentref_/9) --corpus2 [grimm_clean.txt](http://_vscodecontentref_/10)
```

## Example Output
The output of the language model includes both generated text and statistical comparison visualizations:

1. Sample output from the text generation function:

    > the others shiver below by , who could not drink , and good , " how many had finished both father , and rest , and so the maid : but then it , all the door , and the garden , i never heard this way out .
    >
2. Sample output from the probability comparison function:
    ![Histogram of Top 50 Most Frequesnt Words in Corpus](https://i.imgur.com/TRi8MDW.png)

    The histogram visualizations are generated using the `matplotlib` library. Ensure it is installed to view the output.

## Custom Text Analysis
You can also analyze corpora beyond the provided samples. Either add your own ```.txt``` file **OR** upload the text of a new book:
1. Find the text of a book you want online and download it into a ```.txt``` file.
2. Read the text into a string using: 
    ```python
    with open('path_to_your_file.txt', 'r') as file:
    text = file.read()
    ```
3. Create the proper book format with ```cleanBookData```:
    ```bash
    python cleaned_text = cleanBookData(text)
    ```
4. Save the book into a new file, then run ```loadBook``` on the new file to generate a corpus. 
5. Call functions like ```unigramProb``` or ```bigramProb``` to analyze the new corpus!
