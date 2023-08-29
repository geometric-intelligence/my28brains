# 28 and Me


## 🤖 Installing My28Brains

1. Clone a copy of `my28brains` from source:
```bash
git clone https://github.com/bioshape-lab/my28brains
cd my28brains
```
2. Create an environment with python >= 3.10
```bash
conda create -n my28brains python=3.10
```
3. Install `my28brains` in editable mode (requires pip ≥ 21.3 for [PEP 660](https://peps.python.org/pep-0610/) support):
```bash
pip install -e .[all]
```
4. Install pre-commit hooks:
```bash
pre-commit install
```