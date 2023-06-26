# LTN on Clustering

## Data
To generate your own pca features: 
1. Download the [gene expression cancer RNA-Seq Data Set](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq) from the UCI machine learning datasets repository
2. Unzip the files to the `TCGA-PANCAN-HiSeq-801x20531/` folder. 
3. Run the preprocess script to obtain the pca features for every point.

Much of the preprocessing is inspired from the [realpython.com tutorial on k-means](https://realpython.com/k-means-clustering-python/).

## Run LTN

Run the training script of one of the LTN configurations, e.g. `python script-log-ltn.py`.