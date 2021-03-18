# Creating feature clusters


library(tsfeatures)
library(tidyverse)
library(factoextra) # For elbow method
library(ClusterR) # Optimal number of clusters
library(LICORS) # kmeans++ algorithm


#Xmeans Clustering
library(RWeka)
WPM("refresh-cache") # Build Weka package metadata cache
WPM("install-package", "XMeans") # Install XMeans package if not previously installed


set.seed(1)
BASE_DIR <- "Localised_Ensembles/datasets/text_data/kaggle_web_traffic/"
OUTPUT_DIR <- "Localised_Ensembles/datasets/text_data/kaggle_web_traffic/clusters/"
filePath <- paste(BASE_DIR,"kaggle_web_traffic_dataset.txt",sep="")
outputFileStartName <- "kaggle_web_traffic_clusters_"
kpp_outputFileStartName <- "kaggle_web_traffic_kpp_clusters_"
xmeans_outputFileStartName <- "kaggle_web_traffic_xmeans_clusters_"
seed_outputFileStartName <- "kaggle_web_traffic_seed_clusters_"
seed_kpp_outputFileStartName <- "kaggle_web_traffic_seed_kpp_clusters_"
maxLength <- 550
required_num_of_clusters <- c(2,3,4,5,6,7)


getIndices <- function(foundClusters, nameVals){
  allIndices <- list()
  
  for(i in 1:length(nameVals)){
    temp <- c()
    for(j in 1:length(foundClusters)){
      if(foundClusters[j]==nameVals[i]){
        temp <- c(temp,j-1)
      }
    }
    allIndices[[i]] <- temp
  }
  
  allIndices
}


data <- read.csv(filePath, col.names = paste0("V",seq_len(maxLength)), header=FALSE)
data <- t(data)

hwl <- bind_cols(
  tsfeatures(data, c("acf_features","entropy","lumpiness","flat_spots","crossing_points")),
  tsfeatures(data,"stl_features", s.window='periodic', robust=TRUE),
  tsfeatures(data, "max_kl_shift", width=48), 
  tsfeatures(data, c("mean","var"), scale=FALSE, na.rm=TRUE),
  tsfeatures(data, c("max_level_shift","max_var_shift"), trim=TRUE)
  
)

features <- select(hwl, mean, var, x_acf1, trend, linearity, curvature, entropy, lumpiness, spike, max_level_shift, max_var_shift, flat_spots, crossing_points, max_kl_shift, time_kl_shift)

#Apply min-max normalization for each column
for(i in 1:ncol(features)){
  features[,i] <- (features[,i]-min(features[,i]))/(max(features[,i])-min(features[,i]))
}

#Kmeans Clustering
for(i in required_num_of_clusters){
  clusters <- kmeans(features, i)
  clusterIndices <- getIndices(clusters$cluster, 1:i)
  filePath <- paste(OUTPUT_DIR,outputFileStartName,i,'.txt',sep="")
  lapply(clusterIndices, write, filePath, append=TRUE, ncolumns=10000)
}


#kmeans++ clustering
for(i in required_num_of_clusters){
  kmeanspp_clusters <- kmeanspp(features, k = i, start = "random", iter.max = 100)
  kmeanspp_clusterIndices <- getIndices(kmeanspp_clusters$cluster, 1:i)
  kpp_filePath <- paste(OUTPUT_DIR,kpp_outputFileStartName,i,'.txt',sep="")
  lapply(kmeanspp_clusterIndices, write, kpp_filePath, append=TRUE, ncolumns=10000)
}

#xmeans clustering
weka_ctrl <- Weka_control( # Create a Weka control object to specify our parameters
  I = 100, # max no iterations overall
  M = 100, # max no iterations in the kmeans loop
  L = 2,   # min no clusters
  H = nrow(features)-2,   # max no clusters
  D = "weka.core.EuclideanDistance", # distance metric
  C = 0.4, S = 1)
x_means <- XMeans(features, control = weka_ctrl)
x_means
optimal_num_of_clusters <- 3 # set this value based on the output of x_means in line 92
xmeans_clusterIndices <- getIndices(as.vector(x_means$class_ids), 0:optimal_num_of_clusters-1)
xmeans_filePath <- paste(OUTPUT_DIR,xmeans_outputFileStartName,optimal_num_of_clusters,'.txt',sep="")
lapply(xmeans_clusterIndices, write, xmeans_filePath, append=TRUE, ncolumns=10000)


#Getting the optimal number of clusters to bes used for kmeans clustering- Elbow method
fviz_nbclust(features, kmeans, method = "wss", k.max = 25) + theme_minimal() + ggtitle("the Elbow Method")   # function to compute total within-cluster sum of squares


#Getting the optimal number of clusters to bes used for kmeans++ clustering- Elbow method
fviz_nbclust(features, kmeanspp, method = "wss", k.max = 25) + theme_minimal() + ggtitle("the Elbow Method")   # function to compute total within-cluster sum of squares

# Set this value accordingly
elbow_optimal_num_of_clusters <- 5

#With random seeds
for(i in required_num_of_clusters){
  set.seed(i)
  seed_clusters <- kmeans(features, elbow_optimal_num_of_clusters)
  seed_clusterIndices <- getIndices(seed_clusters$cluster, 1:elbow_optimal_num_of_clusters)
  seed_filePath <- paste(OUTPUT_DIR,seed_outputFileStartName,i,'.txt',sep="")
  lapply(seed_clusterIndices, write, seed_filePath, append=TRUE, ncolumns=10000)
}


for(i in required_num_of_clusters){
  set.seed(i)
  seed_kpp_clusters <- kmeanspp(features, k = elbow_optimal_num_of_clusters, start = "random", iter.max = 100)
  seed_kpp_clusterIndices <- getIndices(seed_kpp_clusters$cluster, 1:elbow_optimal_num_of_clusters)
  seed_kpp_filePath <- paste(OUTPUT_DIR,seed_kpp_outputFileStartName,i,'.txt',sep="")
  lapply(seed_kpp_clusterIndices, write, seed_kpp_filePath, append=TRUE, ncolumns=10000)
}



