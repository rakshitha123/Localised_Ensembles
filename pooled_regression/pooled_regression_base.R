# Pooled Regression - basic functions

library(glmnet)
address_near_zero_instability = 0

# fit and forecast from a normal model
fit_normal_model = function(fitting_data, lag, final_lags, horizon, output_file_name, series_means) {
  # create the formula
  no_of_predictors = ncol(fitting_data) - 1
  formula = "y ~ "
  for(predictor in 1:no_of_predictors){
    if(predictor != no_of_predictors){
      formula = paste(formula, "Lag", predictor, " + ", sep="") 
    }else{
      formula = paste(formula, "Lag", predictor, sep="") 
    }
  }
  
  formula = paste(formula, "+ 0", sep="")
  formula = as.formula(formula)
  
  # fit the model
  model = glm(formula = formula, data=fitting_data)
  
  # do the forecasting
  forec_recursive(lag, model, final_lags, horizon, output_file_name, series_means)
}


#recursive forecasting of the series until a given horizon
#for series without mean normalization, the default value for series_means is one
forec_recursive = function(lag, model, final_lags, horizon, output_file_name, series_means) {
  # recursively predict for all the series until the forecast horizon
  predictions = NULL
  for (i in 1:horizon){
    # glm requires a dataframe and and glmnet requires a matrix
    if(model$call[[1]] == "glm"){
      final_lags = as.data.frame(final_lags)
      new_predictions = predict.glm(object=model, newdata = final_lags)  
    }else{
      new_predictions = predict.glmnet(object = model, newx = final_lags) 
    }
    predictions = cbind(predictions, new_predictions)
    
    # update the final lags
    final_lags = as.matrix(final_lags[,1:(ncol(final_lags) - 1)])
    if(dim(final_lags)[2] == 1){
      final_lags = t(final_lags)
    }
    
    final_lags = cbind(new_predictions, final_lags)
    colnames(final_lags) = paste("Lag", 1:lag, sep="")
  }
  # renormalize the predictions
  true_predictions = predictions * as.vector(series_means)
  
  # write true predictions to file
  write.table(true_predictions, file=output_file_name, row.names = F, col.names=F, sep=",", quote=F)
}


