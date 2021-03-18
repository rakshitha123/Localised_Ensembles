# Hyperparameter tuning of Feed-Forward Neural Networks 

import numpy as np
import subprocess
import random

from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from configs.global_configs import hyperparameter_tuning_configs
from utility_scripts.persist_optimized_config_results import persist_results


base_dir = 'Localised_Ensembles'
optimized_config_directory = 'results/optimized_configurations/ffnn_optimisation/'


#Change these parameters for different datasets
original_file = "datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt"
results_file = "datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_results.txt"
horizon = 59
input_window_size = 9
address_near_zero_insability = 1
integer_conversion = 1
seasonality_period = 7
dataset_name = "kaggle_web_traffic"
cluster_file_path = "datasets/text_data/kaggle_web_traffic/clusters/kaggle_web_traffic_dtw_clusters_"
with_different_seeds = 0     #For seed clustering
cluster_number_for_seeds = 0
optimal_num_of_clusters = 0   #For non-ensemble clustering

#Change these paremeters if it requires hyperparameter tuning
require_validation = False
max_hidden_nodes = 12
num_of_series = 997

# Change these paremeters if it does not require hyperparameter tuning
num_of_hidden_nodes = [11, 8]
decay = [0.003914607688071137, 0.036485481443531476]


def smac():
    # Build Configuration Space which defines all parameters and their ranges
    configuration_space = ConfigurationSpace()

    #Define initial ranges
    num_of_hidden_nodes = UniformIntegerHyperparameter("num_of_hidden_nodes", 1, max_hidden_nodes, default_value=1)
    decay = UniformFloatHyperparameter("decay", 0, 0.1, default_value=0)

    configuration_space.add_hyperparameters([num_of_hidden_nodes, decay])

    # creating the scenario object
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": hyperparameter_tuning_configs.SMAC_RUNCOUNT_LIMIT,
        "cs": configuration_space,
        "deterministic": "true",
        "abort_on_first_run_crash": "false"
    })

    # optimize using an SMAC object
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(1), tae_runner=train_model)

    incumbent = smac.optimize()
    return incumbent.get_dictionary()


def train_model(configs):
    num_of_hidden_nodes = configs["num_of_hidden_nodes"]
    decay = configs["decay"]
    print(configs)
    out = subprocess.check_output(["Rscript", "--vanilla", base_dir + "feed_forward_nn/ffnn_optimiser.R", str(num_of_hidden_nodes), str(decay), original_file, dataset_name, str(horizon), str(current_lag), str(address_near_zero_insability), str(integer_conversion), str(1), str(chosen_indices)], stderr = subprocess.STDOUT)
    out = out.split()
    error = float(out[len(out)-1])
    print(error)
    return error


if __name__ == '__main__':
    # Define 2 lags for each dataset
    lags = [input_window_size, 10]

    count = 0
    for lag in lags:
        current_lag = lag

        if(require_validation):
            range_of_series = [i for i in range(0, num_of_series)]

            # Randomly choose 1/7 series and tune hyperparameters for that
            chosen_indices = random.sample(range_of_series, int(len(range_of_series) * (1 / 7)))

            optimized_configuration = smac()

            # persist the optimized configuration to a file
            persist_results(optimized_configuration, base_dir + optimized_config_directory + "fnn_" + dataset_name + '_lag_'+ str(lag) + '.txt')
            validation_error = train_model(optimized_configuration)
            print(optimized_configuration)
            print(validation_error)
        else:          
            # Calculate forecasts FFNN forecasts for the original dataset using a chosen clustering approach
            subprocess.call(["Rscript", "--vanilla", base_dir + "feed_forward_nn/ffnn_cluster_tester.R",
                                 str(num_of_hidden_nodes[count]),
                                 str(decay[count]),
                                 original_file,
                                 results_file,
                                 dataset_name,
                                 str(horizon),
                                 str(lag),
                                 str(address_near_zero_insability),
                                 str(integer_conversion),
                                 str(seasonality_period),
                                 cluster_file_path,
                                 str(with_different_seeds),
                                 str(cluster_number_for_seeds),
                                 str(optimal_num_of_clusters)])
        count = count + 1



