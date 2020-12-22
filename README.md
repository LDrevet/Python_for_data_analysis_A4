# Python_for_data_analysis_A4

## Song release year prediction

In here we are gonna analyse the One Million Song dataset.
This is a collaboration between LabROSA (Columbia University) and The Echo Nest, 
prepared by T. Bertin-Mahieux

The database can be downloaded from : 
https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD

It consists of 515345 records of songs that were composed during the years 1922-2011. Each record consists of 91 features. The first feature is the year in which the song was composed, and the remaining 90 features are various quantities (float) related to the song audio

## Problematic

- Is there a strong relation between the musical features of a song to the year it was composed?
- Can we design several models that can predict the year from the other 90 musical features?
- If the answer is yes, which one will fit in the better way? 

A positive answer to the second question would reveal a profound insight on the nature of a musical composition

## Data Set Information

According to the database authors, we should respect the following train/test split:

- Training set: first 463,715 examples
- Validation set: last 51,630 examples

It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.

## Attribute Information

Each song is caracteried by 91 variables : the first feature is the year of the released of the song, the target of this dataset. The other 90 features are various timbre features related to the song audio. 


# Model

We manage to build a succesful neural network but we had to regroup the years together (group of 10 years)
