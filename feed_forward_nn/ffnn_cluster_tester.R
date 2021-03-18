library(nnet)
library(smooth)
library(MASS)

BASE_DIR <- "Localised_Ensembles"
START_CLUSTER_NUMBER <- 2
FINISH_CLUSTER_NUMBER <- 7

set.seed(1)

args <- commandArgs(trailingOnly = TRUE)
hidden_nodes <-as.numeric(args[1])
decay <- as.numeric(args[2])
file <- paste0(BASE_DIR, args[3])
results <- read.csv(paste0(BASE_DIR, args[4]), header = FALSE, sep=";")
dataset_name <- args[5]
horizon <- as.numeric(args[6])
lag <- as.numeric(args[7])
address_near_zero_insability <- as.numeric(args[8])
integer_conversion <- as.numeric(args[9])
seasonality_period <- as.numeric(args[10])
cluster_file_path <- paste0(BASE_DIR, args[11])
with_different_seeds <- as.numeric(args[12])
cluster_number_for_seeds <- as.numeric(args[13])
optimal_num_of_clusters <- as.numeric(args[14])


if(optimal_num_of_clusters > 0){
  START_CLUSTER_NUMBER <- optimal_num_of_clusters
  FINISH_CLUSTER_NUMBER <- optimal_num_of_clusters
}


dataset <- readLines(file)
dataset <- strsplit(dataset, ",")
results <- results[-1]
num_of_series <- length(dataset)
mase_vector <- NULL
processed_forecasts_file <- paste0(BASE_DIR, "results/ffnn_forecasts/", dataset_name, "_lag_", lag, ".txt")
SMAPE_file_full_name_all_errors <- paste0(BASE_DIR, "results/errors/all_smape_errors_fnn_",dataset_name,"_lag_",lag,".txt")
MASE_file_full_name_all_errors <- paste0(BASE_DIR, "results/errors/all_mase_errors_fnn_",dataset_name,"_lag_",lag,".txt")


#calculate forecasts for a given cluster
calculate_forecasts <- function(cluster_data) {
  series_means_vector <- NULL
  embedded_series <- NULL
  final_lags <- NULL
  series_means <- NULL
  
  
  for (i in 1:length(cluster_data)) {
    time_series <- unlist(cluster_data[i], use.names = FALSE)
    time_series <- as.numeric(time_series)

    #preprocessing
    mean <- mean(time_series)
    
    series_means <- c(series_means, mean)
    time_series <- time_series / mean
    time_series <- log(time_series+1)
    
    # embed the series
    embedded <- embed(time_series, lag + 1)
    if (!is.null(embedded_series)) {
      embedded_series <- as.matrix(embedded_series)
    }
    embedded_series <- rbind(embedded_series, embedded)
    series_means_vector <- c(series_means_vector, rep(mean, nrow(embedded)))
    
    if (!is.null(final_lags)) {
      final_lags <- as.matrix(final_lags)
    }
    final_lags <- rbind(final_lags, rev(tail(time_series, lag)))
  }
  
  # fit the ffnn model
  embedded_series = as.data.frame(embedded_series)
  colnames(embedded_series)[1] = "y"
  colnames(embedded_series)[2:ncol(embedded_series)] = paste("Lag", 1:lag, sep = "")
  
  final_lags = as.data.frame(final_lags)
  colnames(final_lags) = paste("Lag", 1:lag, sep = "")
  
  cluster_forecasts <- fit_ffnn_model(fitting_data = embedded_series, final_lags, series_means)
  cluster_forecasts
}

# fit and forecast from the ffnn model
fit_ffnn_model = function(fitting_data, final_lags, series_means) {
  # create the formula
  no_of_predictors = ncol(fitting_data) - 1
  formula = "y ~ "
  for (predictor in 1:no_of_predictors) {
    if (predictor != no_of_predictors) {
      formula = paste(formula, "Lag", predictor, " + ", sep = "")
    } else{
      formula = paste(formula, "Lag", predictor, sep = "")
    }
  }
  
  formula = paste(formula, "+ 0", sep = "")
  formula = as.formula(formula)
  
  # fit the model
  model = nnet(formula = formula, data = fitting_data, size = hidden_nodes, decay = decay, linout = TRUE)
  
  # do the forecasting
  predictions = forec_recursive(model, final_lags, series_means)
  predictions
}

#recursive forecasting of the series until a given horizon
forec_recursive = function(model, final_lags, series_means) {
  # recursively predict for all the series until the forecast horizon
  predictions = NULL
  for (i in 1:horizon) {
    final_lags = as.data.frame(final_lags)
    new_predictions = predict(model, final_lags)
    predictions = cbind(predictions, new_predictions)
    
    # update the final lags
    final_lags = as.matrix(final_lags[, 1:(ncol(final_lags) - 1)])
    if (dim(final_lags)[2] == 1) {
      final_lags = t(final_lags)
    }
    
    final_lags = cbind(new_predictions, final_lags)
    colnames(final_lags) = paste("Lag", 1:lag, sep = "")
  }
  # renormalize the predictions
  true_predictions <- exp(predictions)
  true_predictions <- true_predictions - 1
  true_predictions = true_predictions * as.vector(series_means)
  
  if (integer_conversion == 1) {
    true_predictions <- round(true_predictions)
  }
  true_predictions[true_predictions<0] <- 0
  
  true_predictions
}


num_of_iterations <- FINISH_CLUSTER_NUMBER - START_CLUSTER_NUMBER + 1
forecast_mat <- array(rep(0, (num_of_series * num_of_iterations * horizon)), c(num_of_series, num_of_iterations, horizon))
final_forecast_mat <-  array(rep(0, (num_of_series * horizon)), c(num_of_series, horizon))


for (clusters in START_CLUSTER_NUMBER:FINISH_CLUSTER_NUMBER) {
  cluster_full_file_path <- paste0(cluster_file_path, clusters, ".txt")
  current_clusters <- readLines(cluster_full_file_path)
  
  if (with_different_seeds) {
    iter <- cluster_number_for_seeds
  } else{
    iter <- clusters
  }
  
  for (i in 1:iter) {
    processing_cluster <- as.numeric(unlist(strsplit(current_clusters[[i]], " ")))
    processing_cluster <- processing_cluster + 1
    processing_dataset <- dataset[processing_cluster]
    forecasts <- calculate_forecasts(processing_dataset)
    
    count <- 1
    for (f in processing_cluster) {
      forecast_mat[f, (clusters - START_CLUSTER_NUMBER + 1), ] <- as.numeric(forecasts[count, ])
      count <- count + 1
    }
  }
}

for (s in 1:num_of_series) {
  for (h in 1:horizon) {
    final_forecast_mat[s, h] <- mean(forecast_mat[s, , h])
  }
}

final_forecasts <- as.data.frame(final_forecast_mat)

if(address_near_zero_insability == 1){
  epsilon = 0.1
  sum = NULL
  comparator = data.frame(matrix((0.5 + epsilon), nrow = nrow(results), ncol = ncol(results)))
  sum = pmax(comparator, (abs(final_forecasts) + abs(results) + epsilon))
  time_series_wise_SMAPE <- 2*abs(final_forecasts-results)/(sum)
  SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm=TRUE)
  
}else{
  time_series_wise_SMAPE <- 2*abs(final_forecasts-results)/(abs(final_forecasts)+abs(results))
  SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm=TRUE)
}

for (k in 1 : nrow(final_forecasts)) {
  mase_vector[k] = MASE(unlist(results[k,]), unlist(final_forecasts[k,]), mean(abs(diff(as.numeric(unlist(dataset[k])), lag = seasonality_period, differences = 1))))
}

# persisting the converted forecasts
write.matrix(final_forecasts, processed_forecasts_file, sep = ",")

# writing the SMAPE results to file
write.table(SMAPEPerSeries, SMAPE_file_full_name_all_errors, row.names = FALSE, col.names = FALSE)

# writing the MASE results to file
write.table(mase_vector, MASE_file_full_name_all_errors, row.names = FALSE, col.names = FALSE)

print(paste0("Mean SMAPE: ", mean(SMAPEPerSeries)))
print(paste0("Median SMAPE: ", median(SMAPEPerSeries)))
print(paste0("Sd SMAPE: ", sd(SMAPEPerSeries))) 
print(paste0("Mean MASE: ", mean(mase_vector))) 
print(paste0("Median MASE: ", median(mase_vector))) 
print(paste0("Sd MASE: ", sd(mase_vector))) 
  
  
  





