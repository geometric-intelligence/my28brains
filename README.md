# 28 and Me - Shape analysis software for characterization of 3D meshes. #

Official implementation of the paper “Geodesic Regression Characterizes 3D Shape Changes in the Female Brain During Menstruation”.

***[[Paper](https://arxiv.org/abs/2210.01932)] published in ICCV proceedings under [[ICCV Workshop Computer Vision for Automated Medical Diagnosis](https://cvamd2023.github.io/)]***

## 🎤 We are developing AI to transform the field of NeuroImaging and Womens' Brain Health: See our Beginner-Friendly Public Talk at Brass Bear Brewing ##

[![BBB Talk](/images/bbb_thumbnail.png)](https://youtu.be/BsdNQUcwb1M)

## 💥 Poster Coming Soon ##

## 🤖 Installing My28Brains

1. Clone a copy of `my28brains` from source:
```bash
git clone https://github.com/bioshape-lab/my28brains
cd my28brains
```
2. Create an environment with python >= 3.10
```bash
conda env create -n my28brains python=3.10
```
3. Install `my28brains` in editable mode (requires pip ≥ 21.3 for [PEP 660](https://peps.python.org/pep-0610/) support):
```bash
pip install -e .[all]
```
4. Install pre-commit hooks:
```bash
pre-commit install
```

## ⭐️ Overview of Goals ##

## 🌎 Bibtex ##
If this code is useful to your research, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2210.01932,
  doi = {10.48550/ARXIV.2210.01932},

  url = {https://arxiv.org/abs/2210.01932},

  author = {Myers, Adele and Miolane, Nina},

  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {Regression-Based Elastic Metric Learning on Shape Spaces of Elastic Curves},

  publisher = {arXiv},

  year = {2022},

  copyright = {Creative Commons Attribution 4.0 International}
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

Create a new project in Wandb called "TODO".

#### 3. Choose the `main` file that fits your goal. 


#### 4. Specify hyperparameters in default_config.py.

#### 5. Run your `main` file.
For a single run, use the command:
```
python main_xxx.py
```

### 6. 👀 See Results.

You can see all of your runs by logging into the Wandb webpage and looking under your project name "TODO".

## 👩‍🔧 Authors ##
[Adele Myers](https://ahma2017.wixsite.com/adelemyers)

[Nina Miolane](https://www.ninamiolane.com/)
