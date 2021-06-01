## sourcecode + dependencies

all files: python>3.5
`wals_linear_regression.py`: tensorflow, numpy, absl-py; matplotlib, scikit-learn for plotting
`generate_statistics.py`: numpy, matplotlib, absl-py, scipy, scikit-learn, adjustText


## computing infrasctructure

PC (Ubuntu 20.04) with HDD and AMD Ryzen 7 3800x 8-core processor

## average run time
`wals_linear_regression`: \~660 seconds

## number of parameters
`wals_linear_regression`: embedding size * category size * number of languages * number of categories

## evaluation metrics
`wals_linear_regression`: direct comparison with the baseline
