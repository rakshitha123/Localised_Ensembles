# Use with ensemble of experts - Pooled Regression

library(glmnet)

BASE_DIR <- "Localised_Ensembles"

args <- commandArgs(trailingOnly = TRUE)
file <- args[1]
horizon <- as.numeric(args[2])
lag <- as.numeric(args[3])
address_near_zero_insability <- as.numeric(args[4])
integer_conversion <- as.numeric(args[5])
is_validation <- as.numeric(args[6])
indices <- args[7]

dataset <- readLines(paste0(BASE_DIR, file))
dataset <- strsplit(dataset, ",")
indices <- substr(indices,2,nchar(indices)-1)
indices <- gsub("\\s", "", indices)
indices <- as.numeric(unlist(strsplit(indices,",")))
indices <- indices + 1 #as indices are ranging from 0

series_means_vector <- NULL
embedded_series <- NULL
final_lags <- NULL
series_means <- NULL

if(is_validation == 1){
  results <- matrix(nrow = length(dataset), ncol = horizon)
}

# fit and forecast from a normal model
fit_model <- function(fitting_data) {
  # create the formula
  no_of_predictors <- ncol(fitting_data) - 1
  formula <- "y ~ "
  for (predictor in 1:no_of_predictors) {
    if (predictor != no_of_predictors) {
      formula <- paste(formula, "Lag", predictor, " + ", sep = "")
    } else{
      formula <- paste(formula, "Lag", predictor, sep = "")
    }
  }

  formula <- paste(formula, "+ 0", sep = "")
  formula <- as.formula(formula)

  # fit the model
  model <- glm(formula = formula, data = fitting_data)

  # do the forecasting
  predictions <- forec_recursive(model)
  predictions
}


#recursive forecasting of the series until a given horizon
forec_recursive <- function(model) {
  predictions <- NULL
  for (i in 1:horizon) {
    final_lags <- as.data.frame(final_lags)
    new_predictions <- predict(model, final_lags)
    predictions <- cbind(predictions, new_predictions)

    # update the final lags
    final_lags <- as.matrix(final_lags[, 1:(ncol(final_lags) - 1)])
    if (dim(final_lags)[2] == 1) {
      final_lags <- t(final_lags)
    }

    final_lags <- cbind(new_predictions, final_lags)
    colnames(final_lags) <- paste("Lag", 1:lag, sep = "")
  }
  # renormalize the predictions
  true_predictions <- predictions
  true_predictions <- true_predictions * as.vector(series_means)

  if (integer_conversion == 1) {
    true_predictions <- round(true_predictions)
  }

    true_predictions[is.na(true_predictions)] <- 0
    true_predictions[true_predictions<0] <- 0

  true_predictions
}


for (i in 1:length(dataset)) {
  original_time_series <- unlist(dataset[i], use.names = FALSE)
  original_time_series <- as.numeric(original_time_series)
  original_series_length <- length(original_time_series)

  #splitting series for training and validation
  if(is_validation == 1){
     time_series <- original_time_series[1:(original_series_length - horizon)]
     results[i, ] <- as.numeric(original_time_series[(original_series_length - horizon + 1):original_series_length])
  }else{
     time_series <- original_time_series
  }

  #preprocessing
  mean <- mean(time_series)
  series_means <- c(series_means, mean)
  time_series <- time_series / mean

  # embed the series
  if(i %in% indices){
      embedded <- embed(time_series, lag + 1)
      if (!is.null(embedded_series)) {
        embedded_series <- as.matrix(embedded_series)
      }
      embedded_series <- rbind(embedded_series, embedded)
      series_means_vector <- c(series_means_vector, rep(mean, nrow(embedded)))
  }

  if (!is.null(final_lags)) {
    final_lags = as.matrix(final_lags)
  }
  final_lags = rbind(final_lags, rev(tail(time_series, lag)))
}


# fit regression model
embedded_series = as.data.frame(embedded_series)
colnames(embedded_series)[1] = "y"
colnames(embedded_series)[2:ncol(embedded_series)] = paste("Lag", 1:lag, sep = "")

final_lags = as.data.frame(final_lags)
colnames(final_lags) = paste("Lag", 1:lag, sep = "")
forecasts <- fit_model(embedded_series)
forecasts <- as.data.frame(forecasts)

#calculate mean SAMPE for optimisation
if(is_validation == 1){
  results <- as.data.frame(results)

  if(address_near_zero_insability == 1){
      epsilon = 0.1
      sum = NULL
      comparator = data.frame(matrix((0.5 + epsilon), nrow = nrow(results), ncol = ncol(results)))
      sum = pmax(comparator, (abs(forecasts) + abs(results) + epsilon))
      time_series_wise_SMAPE <- 2*abs(forecasts-results)/(sum)
      SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm=TRUE)
  }else{
      time_series_wise_SMAPE <- 2*abs(forecasts-results)/(abs(forecasts)+abs(results))
      SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm=TRUE)
  }

  write(SMAPEPerSeries, stderr())
}else{
  write(as.character(forecasts), stderr())
}
