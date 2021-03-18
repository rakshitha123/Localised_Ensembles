import tensorflow as tf
import numpy as np
import csv
import json
import argparse
import multiprocessing

from ensemble_models.expert_tester import ExpertTester as ExpertTester
from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader
from configs.global_configs import model_testing_configs
from utility_scripts.invoke_r_final_evaluation import invoke_r_script

BASE_DIR = "Localised_Ensembles"
LSTM_USE_PEEPHOLES = True
START_CLUSTER_NUMBER = 2
FINISH_CLUSTER_NUMBER = 7
BIAS = False
listOfAssignmentIndices = []
meta_data_size = 0
numOfSeries = 0


def writeTfRecords(trainPath, validationPath, testPath, binaryTrainPath, binaryValidationPath, binaryTestPath):
    tfrecord_writer = TFRecordWriter(
        input_size = model_kwargs['input_size'],
        output_size = model_kwargs['output_size'],
        train_file_path = trainPath,
        validate_file_path = validationPath,
        test_file_path = testPath,
        binary_train_file_path = binaryTrainPath,
        binary_validation_file_path = binaryValidationPath,
        binary_test_file_path = binaryTestPath
    )

    tfrecord_writer.read_text_data()

    if (binaryTrainPath):
        tfrecord_writer.write_train_data_to_tfrecord_file()
    if (binaryValidationPath):
        tfrecord_writer.write_validation_data_to_tfrecord_file()
    if (binaryTestPath):
        tfrecord_writer.write_test_data_to_tfrecord_file()


def filter_data(original_file_path, filtered_file_path, assignedIndexes):
    original_file = open(original_file_path, "r")
    filtered_file = open(filtered_file_path, "w")

    for line in original_file:
        currentSeriesIndex = int((line.split("|"))[0])
        if (currentSeriesIndex in assignedIndexes):
            filtered_file.write(line)

    original_file.close()
    filtered_file.close()


def getForecasts(files,model_kwargs,return_dict):
    num_of_iterations = FINISH_CLUSTER_NUMBER - START_CLUSTER_NUMBER + 1
    forecast_mat = np.full((numOfSeries, num_of_iterations, model_kwargs['output_size']), 0.0)
    final_forecast_mat =  np.full((numOfSeries, output_size), 0.0)

    for iter in range(START_CLUSTER_NUMBER,(FINISH_CLUSTER_NUMBER+1)):
        listOfAssignmentIndices = fillAssignedIndices(files['cluster_file_path'] + "_"+ str(iter) + ".txt")

        if(with_different_seeds):
            value = cluster_number_for_seeds
        else:
            value = iter

        for net in range(value):
            print("Testing with "+str(iter)+" clusters")
            if(len(listOfAssignmentIndices[net])>0):
                print("Testing net "+str(net))
                tester = ExpertTester(**model_kwargs)
                (listOfAssignmentIndices[net]).sort()

                filter_data(files['validation_file_path'], files['filtered_validation_file_path'], listOfAssignmentIndices[net])
                filter_data(files['test_file_path'], files['filtered_test_file_path'], listOfAssignmentIndices[net])

                writeTfRecords('', files['filtered_validation_file_path'], files['filtered_test_file_path'], '', files['filtered_binary_validation_file_path'], files['filtered_binary_test_file_path'])

                tfrecord_reader = TFRecordReader(model_kwargs['input_size'], model_kwargs['output_size'], meta_data_size)

                training_dataset = tf.data.TFRecordDataset(filenames=[files['filtered_binary_validation_file_path']], compression_type="ZLIB")
                training_dataset = training_dataset.repeat(count=int(model_kwargs['max_epoch_size']))
                training_dataset = training_dataset.map(tfrecord_reader.validation_data_parser)

                test_dataset = tf.data.TFRecordDataset([files['filtered_binary_test_file_path']], compression_type="ZLIB")
                test_dataset = test_dataset.map(tfrecord_reader.test_data_parser)

                # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
                train_padded_shapes = ([], [tf.Dimension(None),model_kwargs['input_size']], [tf.Dimension(None), model_kwargs['output_size']], [tf.Dimension(None), model_kwargs['output_size'] + 1])
                test_padded_shapes = ([], [tf.Dimension(None), model_kwargs['input_size']], [tf.Dimension(None), model_kwargs['output_size'] + 1])

                padded_training_data_batches = training_dataset.padded_batch(batch_size=int(model_kwargs['minibatch_size']), padded_shapes=train_padded_shapes)
                padded_test_input_data = test_dataset.padded_batch(batch_size=int(model_kwargs['minibatch_size']), padded_shapes=test_padded_shapes)

                test_params = {
                    'max_num_epochs': int(round(model_kwargs['max_num_epochs'])),
                    'padded_training_data_batches': padded_training_data_batches,
                    'padded_test_input_data': padded_test_input_data
                }

                p = multiprocessing.Process(target=tester.test_model, args=(test_params, return_dict))
                p.start()
                p.join()
                forecasts = return_dict[1]
                print("forecasts")

                count = 0
                for f in listOfAssignmentIndices[net]:
                    forecast_mat[f,(iter-START_CLUSTER_NUMBER)] = forecasts[count]
                    count = count + 1

    for il in range(numOfSeries):
        for f in range(output_size):
            final_forecast_mat[il, f] = np.mean(forecast_mat[il, :, f])  # averaging forecasts over all networks

    return final_forecast_mat


def fillAssignedIndices(cluster_path):
    listOfAssignmentIndices = []
    cluster_file = open(cluster_path, "r")
    for line in cluster_file:
        listOfAssignmentIndices.append(list(map(int,line.split(" "))))

    return listOfAssignmentIndices


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Clustering")
    argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
    argument_parser.add_argument('--contain_zero_values', required=True, help='Whether the dataset contains zero values(0/1)')
    argument_parser.add_argument('--address_near_zero_instability', required=False, help='Whether to use a custom SMAPE function to address near zero instability(0/1). Default is 0')
    argument_parser.add_argument('--integer_conversion', required=False, help='Whether to convert the final forecasts to integers(0/1). Default is 0')
    argument_parser.add_argument('--num_of_series', required=True, help='Number of series in the original data file')
    argument_parser.add_argument('--output_text_dir', required=True, help='Name of the directory containing txt files')
    argument_parser.add_argument('--output_binary_dir', required=True, help='Name of the directory containing binary files')
    argument_parser.add_argument('--input_file_start_name', required=True, help='Starting name of data files')
    argument_parser.add_argument('--txt_test_file', required=True, help='The txt file for test dataset')
    argument_parser.add_argument('--actual_results_file', required=True, help='The txt file of the actual results')
    argument_parser.add_argument('--original_data_file', required=True, help='The txt file of the original dataset')
    argument_parser.add_argument('--cell_type', required=False, help='The cell type of the RNN(LSTM/GRU/RNN). Default is LSTM')
    argument_parser.add_argument('--input_size', required=False, help='The input size of the moving window. Default is 0')
    argument_parser.add_argument('--forecast_horizon', required=True, help='The forecast horizon of the dataset')
    argument_parser.add_argument('--optimizer', required=True, help='The type of the optimizer(cocob/adam/adagrad...)')
    argument_parser.add_argument('--input_format', required=True, help='Input format(moving_window/non_moving_window)')
    argument_parser.add_argument('--without_stl_decomposition', required=False, help='Whether not to use stl decomposition(0/1). Default is 0')
    argument_parser.add_argument('--with_truncated_backpropagation', required=False, help='Whether not to use truncated backpropagation(0/1). Default is 0')
    argument_parser.add_argument('--with_accumulated_error', required=False, help='Whether to accumulate errors over the moving windows. Default is 0')
    argument_parser.add_argument('--seasonality_period', required=True, help='The seasonality period of the time series')
    argument_parser.add_argument('--hyperparameters', required=True, help='File path containing hyperparameters')
    argument_parser.add_argument('--seed', required=True, help='Integer seed to use as the random seed')
    argument_parser.add_argument('--cluster_file_path', required=True, help='Cluster file path')
    argument_parser.add_argument('--optimal_num_of_clusters', required=False, help='Optimal number of clusters if do not need to perform ensembling')
    argument_parser.add_argument('--with_different_seeds', required=False, help='Whether to get predictions for different seeds')
    argument_parser.add_argument('--cluster_number_for_seeds', required=False, help='Number of clusters if needs to run with different seeds')
    argument_parser.add_argument('--clustering_type', required=False, help='Type of clustering: DTW, k-means for features etc.')


    args = argument_parser.parse_args()

    with open(args.hyperparameters, "r") as read_file:
        optimized_params = json.load(read_file)

    dataset_name = args.dataset_name
    seasonality_period = args.seasonality_period
    contain_zero_values = int(args.contain_zero_values)

    if args.clustering_type:
        clustering_type = args.clustering_type
    else:
        clustering_type = "kmeans_clustering"

    if args.input_size:
        input_size = int(args.input_size)
    else:
        input_size = 0

    output_size = int(args.forecast_horizon)
    optimizer = args.optimizer
    input_format = args.input_format
    seed = int(args.seed)

    output_dir = BASE_DIR + args.output_text_dir
    output_binary_dir = BASE_DIR + args.output_binary_dir

    files = {
        'train_file_path': output_dir + args.input_file_start_name + args.forecast_horizon +"i" + args.input_size + ".txt",
        'filtered_train_file_path': output_dir + "filtered"+ args.input_file_start_name + args.forecast_horizon +"i" + args.input_size + ".txt",
        'filtered_binary_train_file_path': output_binary_dir + "filtered"+ args.input_file_start_name + args.forecast_horizon +"i" + args.input_size + ".tfrecords",
        'validation_file_path': output_dir + args.input_file_start_name + args.forecast_horizon +"i" + args.input_size + "v.txt",
        'binary_validation_file_path': output_binary_dir + args.input_file_start_name + args.forecast_horizon + "i" + args.input_size + "v.tfrecords",
        'filtered_validation_file_path': output_dir + "filtered"+ args.input_file_start_name + args.forecast_horizon +"i" + args.input_size + "v.txt",
        'filtered_binary_validation_file_path': output_binary_dir + "filtered"+ args.input_file_start_name + args.forecast_horizon +"i" + args.input_size + "v.tfrecords",
        'test_file_path': output_dir + args.input_file_start_name + "test_"+ args.forecast_horizon +"i" + args.input_size + ".txt",
        'binary_test_file_path': output_binary_dir + args.input_file_start_name + "test_" + args.forecast_horizon + "i" + args.input_size + ".tfrecords",
        'filtered_test_file_path': output_dir + "filtered" + args.input_file_start_name + "test_" + args.forecast_horizon + "i" + args.input_size + ".txt",
        'filtered_binary_test_file_path': output_binary_dir + "filtered" + args.input_file_start_name + "test_" + args.forecast_horizon + "i" + args.input_size + ".tfrecords",
        'txt_test_file_path' : args.txt_test_file,
        'actual_results_file_path' : args.actual_results_file,
        'original_data_file_path' : args.original_data_file,
        'cluster_file_path': args.cluster_file_path
    }

    if args.without_stl_decomposition:
        without_stl_decomposition = bool(int(args.without_stl_decomposition))
    else:
        without_stl_decomposition = False

    if args.optimal_num_of_clusters:
        START_CLUSTER_NUMBER = int(args.optimal_num_of_clusters)
        FINISH_CLUSTER_NUMBER = int(args.optimal_num_of_clusters)

    if args.with_truncated_backpropagation:
        with_truncated_backpropagation = bool(int(args.with_truncated_backpropagation))
    else:
        with_truncated_backpropagation = False

    if args.cell_type:
        cell_type = args.cell_type
    else:
        cell_type = "LSTM"

    if args.with_accumulated_error:
        with_accumulated_error = bool(int(args.with_accumulated_error))
    else:
        with_accumulated_error = False

    if args.with_different_seeds:
        with_different_seeds = bool(int(args.with_different_seeds))
    else:
        with_different_seeds = False

    if (args.cluster_number_for_seeds):
        cluster_number_for_seeds = int(args.cluster_number_for_seeds)

    if args.address_near_zero_instability:
        address_near_zero_instability = bool(int(args.address_near_zero_instability))
    else:
        address_near_zero_instability = False

    if args.integer_conversion:
        integer_conversion = bool(int(args.integer_conversion))
    else:
        integer_conversion = False
    if with_truncated_backpropagation:
        tbptt_identifier = "with_truncated_backpropagation"
    else:
        tbptt_identifier = "without_truncated_backpropagation"

    if without_stl_decomposition:
        stl_decomposition_identifier = "without_stl_decomposition"
    else:
        stl_decomposition_identifier = "with_stl_decomposition"

    if with_accumulated_error:
        accumulated_error_identifier = "with_accumulated_error"
    else:
        accumulated_error_identifier = "without_accumulated_error"

    model_identifier = dataset_name + "_" + clustering_type + "_" + cell_type + "cell" + "_" + input_format + "_" + stl_decomposition_identifier + "_" + optimizer + "_" + tbptt_identifier + "_" + accumulated_error_identifier + "_" + str(seed)
    numOfSeries = int(args.num_of_series)

    model_kwargs = {
        'use_bias': BIAS,
        'use_peepholes': LSTM_USE_PEEPHOLES,
        'cell_type': cell_type,
        'input_size': input_size,
        'output_size': output_size,
        'seed': seed,
        'without_stl_decomposition': without_stl_decomposition,
        'contain_zero_values': contain_zero_values,
        'integer_conversion': integer_conversion,
        'address_near_zero_instability': address_near_zero_instability,
        'optimizer': optimizer,
        'num_hidden_layers': optimized_params['num_hidden_layers'],
        'cell_dimension': optimized_params['cell_dimension'],
        'l2_regularization': optimized_params['l2_regularization'],
        'gaussian_noise_stdev': optimized_params['gaussian_noise_stdev'],
        'random_normal_initializer_stdev': optimized_params['random_normal_initializer_stdev'],
        'minibatch_size': optimized_params['minibatch_size'],
        'max_epoch_size': optimized_params['max_epoch_size'],
        'max_num_epochs': optimized_params['max_num_epochs'],
        'learning_rate': ''
    }

    if without_stl_decomposition:
        meta_data_size = 1
    else:
        meta_data_size = output_size + 1

    tfrecord_reader = TFRecordReader(input_size, output_size, meta_data_size)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    final_forecasts = getForecasts(files,model_kwargs,return_dict)

    rnn_forecasts_full_file_path = BASE_DIR + model_testing_configs.RNN_FORECASTS_DIRECTORY + model_identifier + '.txt'

    with open(rnn_forecasts_full_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(final_forecasts)

    error_file_name = model_identifier + '.txt'
    rnn_forecasts_file_path = model_testing_configs.RNN_FORECASTS_DIRECTORY + model_identifier + '.txt'
    invoke_r_script(rnn_forecasts_file_path, error_file_name, files['txt_test_file_path'],
                     files['actual_results_file_path'], files['original_data_file_path'], str(input_size), str(output_size),
                     str(contain_zero_values), str(int(address_near_zero_instability)),
                     str(int(integer_conversion)), str(int(seasonality_period)), str(int(without_stl_decomposition)))

    print("finished")



