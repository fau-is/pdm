## Predictive deviation monitoring (pdm)

## General
PDM is an outcome-oriented predictive business process monitoring method for predictive deviation monitoring using conformance checking and deep learning.

## Paper
If you use our code or fragments of it, please cite our paper:

```
@proceedings{weinzierl2020pdm,
    title={Predictive Business Process Deviation Monitoring},
    author={Sven Weinzierl and Sebastian Dunzer and Johannes Tenschert and Sandra Zilker and Martin Matzner},
    booktitle={Proceedings of the 29th European Conference on Information Systems (ECIS2021)},
    year={2021}
}
```

## Setup
   1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) 
   2. After setting up miniconda you can make use of the `conda` command in your command line (e.g. CMD or Bash)
   3. To quickly install the `pdm` package, run `pip install -e .` inside the root directory.
   4. To install required packages run `pip install -r requirements.txt` inside the root directory.
   5. Set parameters in `config.py`
   6. Train and test the Bi-LSTM models for the next activity prediction by executing `runner.py`


