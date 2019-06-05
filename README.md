# NMTF-DrugRepositioning

This repository has been created to present the recent work done on drug repositioning thanks to Non-Negative Matrix Factorization.

The jupyter notebook [***results.ipynb***](results.ipynb) presents these results.

## What can you find in this repository ?

This repository contains all data, scripts and results related to our recent work.
In particular, you will find:
- 2 folders, [***data***](data/) which stores the initial data and [***tmp***](tmp/) which stores the results;
- 3 *.py* files, [***load_data_NMTF.py***](load_data_NMTF.py), [***method_NMTF_DatasetContribution.py***](method_NMTF_DatasetContribution.py) and [***method_NMTF.py***](method_NMTF.py) which create classes used in other files. In particular, the last file contains all methods related to the NMTF;
- 4 other *.py* files, [***DatasetContribution.py***](DatasetContribution.py), [***dispersion4.py***](dispersion4.py), [***improvements.py***](improvements.py) and [***initialization.py***](initializaton.py) which compute the results presented in the jupyter notebook.

## How to run the notebook ?

If you want to run these files, you may need to install the following packages:

*sklearn, matplotlib, tqdm, scipy, numpy, pandas, seaborn, csv, cs, [spherecluster](https://github.com/jasonlaska/spherecluster)*
