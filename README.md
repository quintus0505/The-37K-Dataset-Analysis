# The 37K Dataset Analysis

This is the code for processing the 37K dataset presented in [How do People Type on Mobile Devices? Observations from a Study with 37,000 Volunteers](https://userinterfaces.aalto.fi/typing37k/), currently support computing Word Modified Ratio (WMR) and Auto-Correction Ratio (AC).

For using the code, you need to download the dataset and put the csv files in the `original_data` folder.

For some reason the IKI recorded in the original open_input_test_sections can not be trusted 100%, we also provide function 
computing iki based on the TIMESTAMP

## Running the code

#### functions
+ **--data-cleaning** Generate the cleaned dataset
+ **--analyze** Analyze the dataset and save the visaulization sheets
+ **--visualize** Generate the plots and save them
+ **--visualize-by-edit-distance** split the data by edit distance for visualization
+ **--visualize-by-sentence-length** split the data by reference sentence length for visualization

#### options
+ **--keyboard** The keyboard used by the participants, currently support "Gboard", without his option, the code will process all the keyboards
+ **--auto-correct** Whether to compute the Auto-Correction Ratio (AC)
+ **--os** Which OS is used when typing, supporting Android and iOS

#### metrics (for analyzing and visualization)
+ **--modification** Compute the modification ratio on characters level
+ **--wmr** Compute the Word Modified Ratio (WMR) on word level
+ **--ac** Compute the Auto-Correction Ratio (AC) on word level
+ **--edit-distance** Compute the edit distance on character level

#### other info (for analyzing and visualization)
+ **--age** get the age vs iki
+ **--num** get the number of test section vs iki

#### other helping tools
+ **--filter** Filter the dataset by the modification ratio, need data cleaning first (modify inside code for above or below)

### Generating the needed dataset

```python
python main.py --data-cleaning --auto-correct --keyboard Gboard

```

### Filtering the dataset

```python
python main.py --filter 0.8 --auto-correct --keyboard Gboard

```

### Analyzing the dataset

```python
python main.py --analyze --keyboard Gboard --auto-correct --modification --wmr --ac

```

### Generating the plots

```python

python main.py --visualize --keyboard Gboard --auto-correct --modification --wmr --ac

```