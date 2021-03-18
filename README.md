# Localised_Ensembles

This repository contains the experiments of our paper titled, "Ensembles of Localised Models for Time Series Forecasting" which is online available at: [https://arxiv.org/abs/2012.15059](https://arxiv.org/abs/2012.15059).
In this work, we study how ensembleing techniques can be used to solve the localisation issues of Global Forecasting Models (GFM). In particular, we use three ensembling localisation techniques: clustering, ensemble of specialists and the ensembles between global and local forecasting models for our study and compare their performance. 
We have also proposed a new methodology of clustered ensembles where we train multiple GFMs on different clusters of series, obtained by changing the number of clusters and cluster seeds.
All experiments have been conducted using three base GFMs namely Recurrent Neural Networks (RNN), Feed-Forward Neural Networks (FFNN) and Pooled Regression (PR) models.


# Experimental Datasets
The experimental weekly datasets are available in the Google Drive folder at this [link](https://drive.google.com/drive/folders/109-ZYZAHQU1YLQfVLDnpgT4MRX_CqINH?usp=sharing)


# Citing Our Work
When using this repository, please cite:

```{r} 
@misc{godahewa2021ensembles,
  title = {Ensembles of Localised Models for Time Series Forecasting},
  author = {Godahewa, Rakshitha and Bandara, Kasun and  Webb, Geoffrey I. and Smyl, Slawek and Bergmeir, Christoph},
  howPublished = {https://arxiv.org/abs/2012.15059},
  year = {2021}
}
```
