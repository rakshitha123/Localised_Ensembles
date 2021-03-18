# Feed-Forward Neural Network - Ensemble of Experts

import numpy as np
import csv
import re
import copy
import subprocess

from configs.global_configs import model_testing_configs, model_training_configs

BASE_DIR = "Localised_Ensembles"
TOP_N = 2
MAX_NUM_OF_ITERATIONS = 15
listOfAssignmentIndices = []
numOfSeries = 0


def get_forecasts(final_rank_mat, NUM_OF_EXPERTS, lag, num_of_hidden_nodes, decay):
    forecast_mat = np.full((numOfSeries, NUM_OF_EXPERTS, forecast_horizon), 0.0)
    final_forecast_mat =  np.full((numOfSeries, forecast_horizon), 0.0)

    for model in range(NUM_OF_EXPERTS):
        if(len(listOfAssignmentIndices[model])>0):
            print("Testing model "+str(model))
            (listOfAssignmentIndices[model]).sort()

            forecasts = get_ffnn_forecasts(listOfAssignmentIndices[model], lag, num_of_hidden_nodes, decay)
            print("forecasts")

            for f in range(numOfSeries):
                forecast_mat[f,model] = forecasts[f]

    for il in range(numOfSeries):
        required_models = final_rank_mat[il,]
        for f in range(forecast_horizon):
           total = 0.0
           for rmodel in required_models:
               total = total + forecast_mat[il,rmodel,f]
           final_forecast_mat[il, f] = total/TOP_N

    return final_forecast_mat


def get_ffnn_forecasts(indices, current_lag, num_of_hidden_nodes, decay):
    out = subprocess.check_output(
        ["Rscript", "--vanilla", BASE_DIR + "feed_forward_nn/ffnn.R", original_data_file, str(forecast_horizon),
         str(current_lag), str(address_near_zero_instability), str(integer_conversion), str(0), str(list(indices)), str(num_of_hidden_nodes), str(decay)], stderr=subprocess.STDOUT)

    out = out.split()
    converted_vals = []

    for ele in out:
        try:
            ele = str(ele)
            ele = re.sub('[bc)(,\']','',ele)
            converted_ele = float(ele)
            converted_vals.append(converted_ele)
        except:
            print("ignored a str value")

    model_forecasts = np.array(converted_vals)
    model_forecasts = model_forecasts.reshape(forecast_horizon, numOfSeries)
    model_forecasts = model_forecasts.transpose()
    return model_forecasts


def train_model(indices, current_lag, num_of_hidden_nodes, decay):
    out = subprocess.check_output(["Rscript", "--vanilla", BASE_DIR + "feed_forward_nn/ffnn.R", original_data_file, str(forecast_horizon),
         str(current_lag), str(address_near_zero_instability), str(integer_conversion), str(1), str(list(indices)), str(num_of_hidden_nodes), str(decay)], stderr=subprocess.STDOUT)
    out = out.split()
    out = out[-numOfSeries:]
    model_error = [float(i) for i in out]
    print(model_error)
    return model_error


if __name__ == '__main__':
    print("Ensemble of Experts with FFNN")

    dataset_name = "kaggle_web_traffic"
    original_data_file = "datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt"
    actual_results_file = "datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_results.txt"
    input_size = 9
    forecast_horizon = 59
    seasonality_period = 7
    address_near_zero_instability = 1
    integer_conversion = 1
    numOfSeries = 997
    NUM_OF_EXPERTS = 5
    hidden_nodes = [11, 8] # Tuned values for hidden nodes
    decays = [0.003914607688071137, 0.036485481443531476] # Tuned values for decay

    is_early_stopping = 1
    lags = [input_size, 10]
    rangeOfSeries = [i for i in range(0, numOfSeries)]  # Start series from 0

    lag_count = 0

    for lag in lags:
        model_identifier = dataset_name + "_" + str(NUM_OF_EXPERTS) + "_experts_lag_" + str(lag) + ".txt"
        perf_mat = np.full((numOfSeries, NUM_OF_EXPERTS), 1e300)
        listOfBestIndices = NUM_OF_EXPERTS * [[0]]
        validationErrors = np.full((MAX_NUM_OF_ITERATIONS,NUM_OF_EXPERTS), 0.0)
        badModels = []

        for model in range(NUM_OF_EXPERTS):
            listOfAssignmentIndices.append(np.random.choice(rangeOfSeries, int(len(rangeOfSeries)*0.5)))

        for model in range(NUM_OF_EXPERTS):
            listOfAssignmentIndices[model] = list(set(listOfAssignmentIndices[model])) #get unique indices

        for iter in range(MAX_NUM_OF_ITERATIONS):
            for model in range(NUM_OF_EXPERTS):
                if ((not is_early_stopping) or (is_early_stopping and (not (model in badModels)))):
                    print("Training model "+str(model)+" with iteration "+str(iter))
                    (listOfAssignmentIndices[model]).sort()

                    smape_errors =  train_model(listOfAssignmentIndices[model], lag, hidden_nodes[lag_count], decays[lag_count])
                    #print(smape_errors)
                    validationErrors[iter,model] =  np.mean(smape_errors)

                    if(is_early_stopping):
                        if((iter==0) or ((iter>0) and (validationErrors[(iter-1),model] > validationErrors[iter,model]))):
                            count = 0
                            for error in smape_errors:
                                perf_mat[count, model] = error
                                count = count + 1
                    else:
                        count = 0
                        for error in smape_errors:
                            perf_mat[count, model] = error
                            count = count + 1

                    if(is_early_stopping):
                        if((iter>0) and (validationErrors[(iter-1),model] < validationErrors[iter,model])):
                            badModels.append(model)
                            listOfAssignmentIndices[model] = previousIndices[model]


            print(validationErrors)

            if(is_early_stopping):
                previousIndices = copy.deepcopy(listOfAssignmentIndices)

            if(is_early_stopping and (len(badModels) > 0)):
                print("Validation error is growing for "+str(badModels))

            if(is_early_stopping and (len(badModels)==NUM_OF_EXPERTS)):
                break

            rank_mat = np.full((numOfSeries, NUM_OF_EXPERTS), -1)

            for il in range(numOfSeries):
                rank_mat[il,] = np.argsort(perf_mat[il,])

            bestAssignments = rank_mat[:, 0];

            for model in range(NUM_OF_EXPERTS):
                if ((not is_early_stopping) or (is_early_stopping and (not (model in badModels)))):
                    listOfBestIndices[model] = np.where(bestAssignments == model)[0]

            for model in range(NUM_OF_EXPERTS):
                if ((not is_early_stopping) or (is_early_stopping and (not (model in badModels)))):
                    listOfAssignmentIndices[model] = listOfBestIndices[model]
                    (listOfAssignmentIndices[model]).sort()
                    for itn in range(1, TOP_N):
                        assignments = rank_mat[:, itn]
                        otherBestIndices = np.where(assignments == model)[0]
                        listOfAssignmentIndices[model] = np.concatenate((listOfAssignmentIndices[model], otherBestIndices))
                        (listOfAssignmentIndices[model]).sort()


                    if ((len(listOfBestIndices[model]) == 0) and (iter!=(MAX_NUM_OF_ITERATIONS-1))) :
                        print("restarting net ", model)
                        listOfAssignmentIndices[model] = np.random.choice(rangeOfSeries, int(len(rangeOfSeries) * 0.5))
                        listOfAssignmentIndices[model] = list(set(listOfAssignmentIndices[model]))
                        (listOfAssignmentIndices[model]).sort()


        print(listOfAssignmentIndices)
        final_rank_mat = np.full((numOfSeries, TOP_N), -1)
        for il in range(numOfSeries):
            rank_mat[il,] = np.argsort(perf_mat[il,])
            for itn in range(TOP_N):
                final_rank_mat[:,itn] = rank_mat[:, itn]

        final_forecasts = get_forecasts(final_rank_mat,NUM_OF_EXPERTS, lag, hidden_nodes[lag_count], decays[lag_count])

        ffnn_forecasts_full_file_path = BASE_DIR + model_testing_configs.FFNN_FORECASTS_DIRECTORY + model_identifier
        validation_error_file_path = BASE_DIR + model_training_configs.VALIDATION_ERRORS_DIRECTORY + "fnn_" + model_identifier

        with open(ffnn_forecasts_full_file_path, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(final_forecasts)

        with open(validation_error_file_path, "w") as output2:
            writer = csv.writer(output2, lineterminator='\n')
            writer.writerows(validationErrors)

        ffnn_forecasts_file_path = model_testing_configs.FFNN_FORECASTS_DIRECTORY + model_identifier

        subprocess.call(
            ["Rscript", "--vanilla", BASE_DIR + "error_calculator/other_model_evaluation.R", dataset_name, ffnn_forecasts_file_path, actual_results_file, original_data_file,
             "ffnn_ensemble_of_experts", str(address_near_zero_instability), str(seasonality_period), str(lag), str(integer_conversion)],
            stderr=subprocess.STDOUT)

        print("finished_lag ",lag)
        lag_count = lag_count + 1

    print("finished all")


