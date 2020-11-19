# panda

This repo contains code to train classification networks to predict the severity of prostate cancer from microscopy scans, as part of the [Kaggle PANDA challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data). 

## Dataset
The challenge dataset contains microscopy scans of prostate biopsy samples at multiple levels of granularity. Scripts to process these whole-slide images are contains in [utils](./utils).

## Models
All models trained correspond to Multiple-Instance learners, where each slide image is treated as a bag-of-patches, from which predictions are made.
