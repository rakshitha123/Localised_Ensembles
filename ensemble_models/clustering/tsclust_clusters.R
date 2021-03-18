# Creating DTW clusters

library(dtwclust)


BASE_DIR <- "Localised_Ensembles/datasets/text_data/kaggle_web_traffic/"
OUTPUT_DIR <- "Localised_Ensembles/datasets/text_data/kaggle_web_traffic/clusters/"
filePath <- paste(BASE_DIR,"kaggle_web_traffic_dataset.txt",sep="")
outputFileStartName <- "kaggle_web_traffic_dtw_clusters_"
maxLength <- 550
containZeroValues <- 1
required_num_of_clusters <- c(2,3,4,5,6,7)  


time_series_list <- list()


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

for(i in 1:ncol(data)){
  current_series <- data[,i]
  current_series <- current_series[!is.na(current_series)]
  current_series <- current_series/mean(current_series)
  if(containZeroValues){
    current_series <- current_series + 1
  }
  current_series <- log(current_series)
  time_series_list[[i]] <- current_series
}

for(i in required_num_of_clusters){
  dtw_clusters <- tsclust(time_series_list, k = i, type="partitional", distance = "dtw", centroid = "pam", seed = 1, trace = TRUE, control = partitional_control(nrep = 1L))
  clusterIndices <- getIndices(dtw_clusters@cluster, 1:i)
  filePath <- paste(OUTPUT_DIR,outputFileStartName,i,'.txt',sep="")
  lapply(clusterIndices, write, filePath, append=TRUE, ncolumns=10000)
}





