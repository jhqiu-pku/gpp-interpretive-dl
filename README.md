## Large contribution of antecedent climate to ecosystem productivity anomalies during extreme events

This repository contains the supporting code for the paper:

> Qiu *et al.* (2025), **Large contribution of antecedent climate to ecosystem productivity anomalies during extreme events**, *Nature Geoscience*, https://doi.org/10.1038/s41561-025-01856-4

### Overview

This study investigates the current and lagged effects of climatic variations on ecosystem productivity. The main codes used for model training and interpretation are provided in this repository.

The repository is structured as follows:                                    
```
|- corecode/                              # core functions and model structure
|   |- dataset.py
|   |- datautils.py
|   |- ealstm.py
|- main.py                  # Main python file used for model training and interpretation
```

###  System requirements
- operating systems: windows 11
- software version: Python 3.11.5

###  Installation guide
Install dependencies:
`pip install -r requirements.txt`

### Instructions to run on data

- Train the model using meteorological and static data (*model_train function*)

- Interpret the trained model for each specific event (*model_interpret_global function*)

### Expected output
- Trained model weights

- Temporal contributions of meteorological variables
