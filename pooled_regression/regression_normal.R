BASE_DIR <- "Localised_Ensembles"

source(paste0(BASE_DIR, "pooled_regression/pooled_regression_base.R"))
source(paste0(BASE_DIR, "error_calculator/base_error_calculator.R"))


file = paste0(BASE_DIR, "datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt")
dataset_name = "kaggle_web_traffic"
horizon = 59
input_window_size <- 9
address_near_zero_insability <- 1
seasonality_period <- 7
actual_forecasts_file = "datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_results.txt"


# for the different lags
lags = c(input_window_size, 10)  

dataset <- readLines(file)
dataset <- strsplit(dataset, ",")


# loop for the different lags
for (lag in lags) {
  series_means_vector = NULL
  embedded_series = NULL
  final_lags = NULL
  series_means = NULL
  
  
  output_file_name_normal = paste0(BASE_DIR,"results/pooled_regression_forecasts/", dataset_name, "_pooled_regression_", lag, "_forecasts.txt")
  unlink(output_file_name_normal)
  
  model_results = NULL
  
  for (i in 1:length(dataset)) {
    time_series <- unlist(dataset[i], use.names = FALSE)
    time_series <- as.numeric(time_series)
    
    mean = mean(time_series)
    
    series_means = c(series_means, mean)
    time_series = time_series / mean
    
    # embed the series
    embedded = embed(time_series, lag + 1)
    if (!is.null(embedded_series)) {
      embedded_series = as.matrix(embedded_series)
    }
    embedded_series = rbind(embedded_series, embedded)
    series_means_vector = c(series_means_vector, rep(mean, nrow(embedded)))
    
    if (!is.null(final_lags)) {
      final_lags = as.matrix(final_lags)
    }
    final_lags = rbind(final_lags, rev(tail(time_series, lag)))
  }
  
  
  # fit a normal model
  embedded_series = as.data.frame(embedded_series)
  colnames(embedded_series)[1] = "y"
  colnames(embedded_series)[2:ncol(embedded_series)] = paste("Lag", 1:lag, sep = "")
  
  final_lags = as.data.frame(final_lags)
  colnames(final_lags) = paste("Lag", 1:lag, sep = "")
  fit_normal_model(fitting_data = embedded_series,
                   lag,
                   final_lags,
                   horizon,
                   output_file_name_normal,
                   series_means)
  
  #calculate the errors
  calculate_errors(output_file_name_normal, actual_forecasts_file, file, paste0(dataset_name,"_",lag,"_"), address_near_zero_insability, seasonality_period)
}
