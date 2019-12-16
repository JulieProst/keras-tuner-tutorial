# keras-tuner-tutorial
Hands on tutorial for keras-tuner

This repo aims at introducing hyperparameter tuning through the Keras Tuner library.
It provides a comparison of its different tuners, applied to computer vision through the CIFAR10 dataset.

This is work in progress, all feedback is welcomed.

### Install the project
- Clone the repo
- Create a virtualenv and activate it:
```
virtualenv -p python3 venv
source venv/bin/activate
```
- Install the requirements:
```
pip install requirements.txt
```


### Results

Tasks duration was measured on CPU


### Run the comparison
Available tuners :

- Random Search
- Hyperband
- Bayesian Optimization

```
python tuner_comparison.py
```
