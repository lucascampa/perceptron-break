# Perceptron with early stopping
This repository contains a minimal experiment with the Perceptron algorithm, extended to stop training early once convergence is reached.  
The implementation is based on the version presented in *Machine Learning with PyTorch and Scikit-Learn* by Sebastian Raschka et al., with a small modification to introduce early stopping.

## Project structure
```
├── notebook/
│ └── perceptron_demo.ipynb # Jupyter notebook showing experiments on the Iris dataset
├── perceptron/
│ ├── init.py
│ └── perceptron.py # Perceptron implementation with early stopping tweak
├── requirements.txt # Dependencies for reproducing results
├── LICENSE
└── .gitignore
```

## Environment
- Python **3.9** (tested on **3.9.6**)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/perceptron.git
cd perceptron
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
After installing the environment, you can reproduce the experiment in two ways:

1. **Run the demo notebook**  

Launch Jupyter and open the notebook with the Perceptron example:
```bash
jupyter notebook notebook/perceptron_demo.ipynb
```

2. **Import the Perceptron class in Python**

You can also use the modified Perceptron implementation directly and compare the results:
```python
from perceptron import Perceptron

# Original implementation
ppn = Perceptron(eta=0.1, n_iter=100)
ppn.fit(X_train, y_train)

# Break when reaching convergence
ppn = Perceptron(eta=0.1, n_iter=100)
ppn.fit_break(X_train, y_train)
```

Replace `X_train, y_train` with your own data (e.g., Iris).

Both approaches use the same implementation, located in perceptron/perceptron.py

## References
This project builds upon the following resources:
- **Perceptron implementation**:

  Sebastian Raschka, Yuxi (Hayden) Liu, and Dmytro Dzhulgakov.  
  *Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing, 2022.  
  [GitHub repository](https://github.com/rasbt/machine-learning-book)

- **Dataset**:
  
  Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository.  
  https://doi.org/10.24432/C56C76