# The 37K Dataset Analysis

This is the code for processing the 37K dataset presented in [How do People Type on Mobile Devices? Observations from a Study with 37,000 Volunteers](https://userinterfaces.aalto.fi/typing37k/), currently support computing Word Modified Ratio (WMR) and Auto-Correction Ratio (AC).

For using the code, you need to download the dataset and put the csv files in the `original_data` folder.

## Running the code

#### functions
+ **--data-cleaning** Generate the cleaned dataset
+ **--analyze** Analyze the dataset and save the visaulization sheets
+ **--visualize** Generate the plots and save them

#### options
+ **--keyboard** The keyboard used by the participants, currently support "Gboard", without his option, the code will process all the keyboards
+ **--auto-correct** Whether to compute the Auto-Correction Ratio (AC)

#### metrics (for analyzing and visualization)
+ **--modification** Compute the modification ratio on characters level
+ **--wmr** Compute the Word Modified Ratio (WMR) on word level
+ **--ac** Compute the Auto-Correction Ratio (AC) on word level

#### other info (for analyzing and visualization)
+ **--age** get the age vs iki

### Generating the needed dataset

```python
python main.py --data-cleaning --auto-correct --keyboard Gboard

```

### Analyzing the dataset

```python
python main.py --analyze --keyboard Gboard --auto-correct --modification --wmr --ac

```

### Generating the plots

```python

python main.py --visualize --keyboard Gboard --auto-correct --modification --wmr --ac

```