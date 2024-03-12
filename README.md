# The 37K Dataset Analysis

This is the code for processing the 37K dataset presented in [How do People Type on Mobile Devices? Observations from a Study with 37,000 Volunteers](https://userinterfaces.aalto.fi/typing37k/), currently support computing Word Modified Ratio (WMR) and Auto-Correction Ratio (AC).

## Running the code

### Generating the needed dataset

```python
python main.py --data-cleaning --auto-correct --keyboard Gboard

```

### Analyzing the dataset

```python
python main.py --analyze --keyboard Gboard

```

### Generating the plots

```python

python main.py --visualize --keyboard Gboard

```