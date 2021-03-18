# Error calculation of Feed-Forward Neural Networks and Pooled Regression models


library(smooth)

base_dir <- "Localised_Ensembles"
errors_dir <- paste(base_dir, "results/ensemble_errors/", sep = "")

args <- commandArgs(trailingOnly = TRUE)
dataset_name <- args[1]
forecast_file <- args[2]
actual_forecasts_file <- args[3]
original_data_file <- args[4]
technique <- args[5]
address_near_zero_insability <- as.numeric(args[6])
seasonality_period <- as.numeric(args[7])
lag <- args[8]
integer_conversion <- as.numeric(args[9])


actual_forecasts_file <- paste(base_dir, actual_forecasts_file, sep = "")
original_data_file <- paste(base_dir, original_data_file, sep = "")
forecast_file <- paste(base_dir, forecast_file, sep = "")

SMAPE_file_full_name_all_errors <- paste0(errors_dir, "all_smape_errors_", dataset_name, "_", technique, "_lag_", lag, ".txt")
MASE_file_full_name_all_errors <- paste0(errors_dir, "all_mase_errors_", dataset_name, "_", technique, "_lag_", lag, ".txt")

# read the forecasts
forecasts_df = read.csv(forecast_file, header = F, sep = ",")
forecasts_df[forecasts_df < 0] <- 0

if(integer_conversion==1){
  forecasts_df <- round(forecasts_df)
}

# read the actual forecasts
actual_forecasts_df <- read.csv(file = actual_forecasts_file, sep = ';', header = FALSE)
actual_forecasts_df <- actual_forecasts_df[, -1]

# reading the original data to calculate the MASE errors
original_dataset <- readLines(original_data_file)
original_dataset <- strsplit(original_dataset, ',')

# calculating the SMAPE
if (address_near_zero_insability == 1) {
  # define the custom smape function
  epsilon = 0.1
  sum = NULL
  comparator = data.frame(matrix((0.5 + epsilon), nrow = nrow(actual_forecasts_df), ncol = ncol(actual_forecasts_df)))
  sum = pmax(comparator, (abs(forecasts_df) + abs(actual_forecasts_df) + epsilon))
  time_series_wise_SMAPE <- 2 * abs(forecasts_df - actual_forecasts_df) / (sum)
  SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm = TRUE)
} else{
  time_series_wise_SMAPE <- 2 * abs(forecasts_df - actual_forecasts_df) / (abs(forecasts_df) + abs(actual_forecasts_df))
  SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm = TRUE)
}

mean_SMAPE = mean(SMAPEPerSeries)
median_SMAPE = median(SMAPEPerSeries)
std_SMAPE = sd(SMAPEPerSeries)

mean_SMAPE = paste("mean_SMAPE", mean_SMAPE, sep = ":")
median_SMAPE = paste("median_SMAPE", median_SMAPE, sep = ":")
std_SMAPE = paste("std_SMAPE", std_SMAPE, sep = ":")

# MASE
mase_vector = NULL

for (k in 1:nrow(forecasts_df)) {
  mase_vector[k] = MASE(unlist(actual_forecasts_df[k, ]), unlist(forecasts_df[k, ]), mean(abs(diff(as.numeric(unlist(original_dataset[k])), lag = seasonality_period, differences = 1))))
}

mean_MASE = mean(mase_vector)
median_MASE = median(mase_vector)
std_MASE = sd(mase_vector)

mean_MASE = paste("mean_MASE", mean_MASE, sep = ":")
median_MASE = paste("median_MASE", median_MASE, sep = ":")
std_MASE = paste("std_MASE", std_MASE, sep = ":")

# writing the SMAPE results to file
write.table(SMAPEPerSeries, SMAPE_file_full_name_all_errors, row.names = FALSE, col.names = FALSE)

# writing the MASE results to file
write.table(mase_vector, MASE_file_full_name_all_errors, row.names = FALSE, col.names = FALSE)

print(mean_SMAPE)
print(median_SMAPE)
print(std_SMAPE)
print(mean_MASE)
print(median_MASE)
print(std_MASE)
