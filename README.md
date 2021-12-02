# Localised_Ensembles

This repository contains the experiments of our paper titled, "Ensembles of Localised Models for Time Series Forecasting" which is published in the [Konwledge-Based Systems](https://doi.org/10.1016/j.knosys.2021.107518) journal.
In this work, we study how ensembleing techniques can be used to solve the localisation issues of Global Forecasting Models (GFM). In particular, we use three ensembling localisation techniques: clustering, ensemble of specialists and the ensembles between global and local forecasting models for our study and compare their performance. 
We have also proposed a new methodology of clustered ensembles where we train multiple GFMs on different clusters of series, obtained by changing the number of clusters and cluster seeds.
All experiments have been conducted using three base GFMs namely Recurrent Neural Networks (RNN), Feed-Forward Neural Networks (FFNN) and Pooled Regression (PR) models.

# Experimental Datasets
The experimental datasets are available in the Google Drive folder at this [link](https://drive.google.com/drive/folders/16xqLEFyLn_gJcXrIp1LWyAD_KvJjB5Hn?usp=sharing). 

Create a folder named "datasets" in the parent level. Create a sub-folder named, "text_data" within the "datasets" folder and place the datasets and the corresponding results files in there.

# Instructions for Execution
## RNN Execution
Our RNN implementations are mainly based on the Tensorflow based framework implemented by Hewamalage et al., 2021.

For RNNs, you need to first preprocess your datasets. For that, use the the R scripts in "./preprocess_scripts" folder. It contains 3 preprocessing R scripts that show the examples of preprocessing for the Kaggle Web Traffic daily dataset. These preprocess scripts basically convert the series into the moving window format to work with RNN models. We generally use 1/7 of time series for hyperparameter tuning. To randomly choose 1/7 time series of the Kaggle Web Traffic dataset, run "./ensemble_models/hyper-parameter-tuning/series-filter-web-traffic.py". For different datasets, change the path variables in the scripts accordingly. After running the R scripts, run the script named as "./preprocess_scripts/create_tfrecords.py" with the parameters corresponding with your dataset to convert the text data into a binary format. The generated tfrecords are used to train the RNN.

For new datasets, add experiments into "./utility_scripts/execution_scripts/rnn_experiments.sh" script according to the format mentioned in the script and run experiments through that. Makesure to provide absolute paths for datasets. This script contains the examples of running the RNN ensemble of experts and the RNN clustering based models. See those examples for more details.

Create a folder named "results" at the parent level and create sub-folders accordingly to store forecasts, errors and optimised parameters.

## FFNN and PR Execution
FFNN implementations are within the "./feed_forward_nn" folder. The Python script, "cluster_optimiser.py" performs all clustering related experiments. Single FFNN related experiments are conducted using the Python script, "optimiser.py". Both scripts also perform hyperparamter tuning. The FFNN ensemble of experts implementation is there within the "./ensemble_models" folder ("ensemble_of_experts_fnn.py").

PR implementations are within the "./pooled_regression" folder. The PR ensemble of experts implementation is there within the "./ensemble_models" folder ("ensemble_of_experts_regression.py").

All script contain examples of running our GFMs for the Kaggle Web Traffic dataset. For different datasets, change the paths and other variables in the scripts accordingly.

# Citing Our Work
When using this repository, please cite:

```{r} 
@article{godahewa_2021_ensembling,
  author = {R. Godahewa and K. Bandara and G. I. Webb and S. Smyl and C. Bergmeir},
  title = {Ensembles of localised models for time series forecasting},
  journal =  {Knowledge-Based Systems},
  volume = {233},
  pages = {107518},
  year = {2021}
}
```

# References
Hewamalage H., Bergmeir C., Bandara K. (2021) Recurrent neural networks for time series forecasting: Current status and future directions. International Journal of Forecasting DOI https://doi.org/10.1016/j.ijforecast.2020.06.008
