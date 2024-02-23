# 28 and Me - Shape analysis software for characterization of 3D meshes. #

Official implementation of the paper “Geodesic Regression Characterizes 3D Shape Changes in the Female Brain During Menstruation”.

***[[Paper](https://arxiv.org/abs/2309.16662)] published in ICCV proceedings under [[ICCV Workshop Computer Vision for Automated Medical Diagnosis](https://cvamd2023.github.io/)]***

## 🎤 We are developing AI to transform the field of NeuroImaging and Womens' Brain Health: See our Beginner-Friendly Public Talk at Brass Bear Brewing ##

[![BBB Talk](/images/bbb_thumbnail.png)](https://youtu.be/BsdNQUcwb1M)
Visual on thumbnail slide taken from: Caitlin M Taylor, Laura Pritschet, and Emily G Jacobs. The scientific body of knowledge–whose body does it serve? A spotlight on oral contraceptives and women’s health factors in neuroimaging. Frontiers in neuroendocrinology, 60:100874, 2021.

## 💥 Poster, Presented at the ICCV Workshop: Computer Vision for Automated Medical Diagnosis ##
![BBB Talk](/images/Adele_Myers_poster.png)

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

## 🌎 Bibtex ##
If this code is useful to your research, please cite:

```
@misc{myers2023geodesic,
      title={Geodesic Regression Characterizes 3D Shape Changes in the Female Brain During Menstruation},
      author={Adele Myers and Caitlin Taylor and Emily Jacobs and Nina Miolane},
      year={2023},
      eprint={2309.16662},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🏃‍♀️ How to Run the Code ##

We use Wandb to keep track of our runs. To launch a new run, follow the steps below.

#### 1. Set up [Wandb](https://wandb.ai/home) logging.

Wandb is a powerful tool for logging performance during training, as well as animation artifacts. To use it, simply [create an account](https://wandb.auth0.com/login?state=hKFo2SBNb0U4SjE0ZWN3OGZtbTlJWTRpYkNmU0dUTWZKSDk3Y6FupWxvZ2luo3RpZNkgODhWd254WW1zdG51RTREd0pWOGVKWVVzZkVOZ0dydGqjY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=dEZVS3dvYXFVSjdjZFFGdw%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true), then run:
```
wandb login
```
to sign into your account.

#### 2. Create a new project in Wandb.

Create a new project in Wandb and give it a project name. Then, edit `main_xxx.py` files so that they use this project name.

#### 3. Specify hyperparameters in default_config.py.

Most Important General Paramters in `default_config.py` (see descriptions in `default_config.py`):

- Elastic Metric Parameters: `a0`, `a1`, `b1`, `c1`, `d1`, `a2`
- `dataset_name`
- `linear_residuals`
- `n_X`

#### 4. Run the `main` file that fits your goal.

- `main_1_preprocess.py`: Preprocesses hippocampus data, as described in the notes at the top of the file.
- `main_2_regression.py`: Runs regression (linear and geodesic) on dataset specified in `default_config.py`. Runs geodesic regression with linear residuals if `linear_residuals = True` in `default_config.py`.
- `main_3_line_vs_geodesic.py`: Creates synthetic data (data type determined by `dataset_name` in `default_config.py`) that lies in a curved space. Then, computes a line between two synthetic data points and a geodesic between the two synthetic data points and compares how different they are, indicating whether the distance between the points is large or small are compared to the curvature of the manifold. Used to inform "rules of thumb" for whether linear regression or geodesic regression with linear residuals can be used to approximate geodesic regression on a particular dataset.

To run one of the main files, use the command:
```
python main_xxx.py
```

where xxx is replaced by the actual name of the file.

### 5. 👀 See Results.

You can see all of your runs by logging into the Wandb webpage and looking under your project name.

## 👩‍🔧 Authors ##
[Adele Myers](https://ahma2017.wixsite.com/adelemyers)

[Nina Miolane](https://www.ninamiolane.com/)

## How to Set up Your Environment

```shell
$ conda create -n my28brains --file conda-linux-64.lock
$ conda activate my28brains
<<<<<<< HEAD
<<<<<<< HEAD
$ poetry install --no-root
```
We use `--no-root` because we don't have a module named `my28brains`

=======
$ poetry install --no-root
```
We use `--no-root` because we don't have a module named `my28brains`
>>>>>>> f17239da1 (Update README.md)
# Dev

Only run if changes are made to the environment files.

To recreate the conda lock, after modifying conda.yaml:
```shell
pip install conda-lock
make conda-linux-64.lock
```

To recreate the poetry lock, after modifying pyproject.toml:
```shell
make poetry.lock
```
