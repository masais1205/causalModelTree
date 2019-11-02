# causalModelTree
Build a causal model tree, an explainable model, for prediction. Use causality-based criteria to grow tree, logistic regression (on causes of the outcome variable) to predict in each leaf node.

# Requirements
pip install -r requirements.txt

# Usage
python CausalModelTree.py -f "filename.csv" -m MT-PC -l 2 -t .2

(-h, see the help messages)
