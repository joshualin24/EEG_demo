# EEG_demo


Note [https://docs.google.com/document/d/1Aonn9KBazC3oPj06o7a4qoS1bSZTf4fO9-azNlpPHq0/edit?usp=sharing]


## Reproduction
To reproduce the environment we used in Colab, create a virtual environment with your favorite choice and run
```
pip install -r requirements.txt
```
, or if you prefer using `conda`, run
```
conda env create -f environment.yaml
conda activate ssleeg
```
to create and activate an environment named `ssleeg`.


## EEGBCI Dataset

| Run | Annotation | Description | Class Label |
| --- | --- | --- | --- |
| 1 | T0 | Baseline, eyes open | 0 |
| 2 | T0 | Baseline, eyes closed | 1 |
| 3, 7, 11 | T1 | Open and close left fist | 2 |
|   | T2 | Open and close right fist | 3 |
| 4, 8, 12 | T1 | Imagine opening and closing left fist | 4 |
|   | T2 | Imagine opening and closing right fist | 5 |
| 5, 9, 13 | T1 | Open and close both fists | 6 |
|   | T2 | Open and close both feet | 7 |
| 6, 10, 14 | T1 | Imagine opening and closing both fists | 8 |
|   | T2 | Imagine opening and closing both feet | 9 |
