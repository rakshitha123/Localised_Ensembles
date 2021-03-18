# Ensemble of experts - RNN

import tensorflow as tf
import numpy as np
import json
import csv
import argparse
import multiprocessing
import copy

from ensemble_models.expert_trainer import ExpertTrainer as ExpertTrainer
from ensemble_models.expert_tester import ExpertTester as ExpertTester
from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader
from configs.global_configs import model_testing_configs, model_training_configs
from utility_scripts.invoke_r_final_evaluation import invoke_r_script

BASE_DIR = "Localised_Ensembles"
LSTM_USE_PEEPHOLES = True
BIAS = False
TOP_N = 2
trainers = []
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


def getForecasts(files,model_kwargs,return_dict,final_rank_mat,NUM_OF_EXPERTS):
    forecast_mat = np.full((numOfSeries, NUM_OF_EXPERTS, model_kwargs['output_size']), 0.0)
    final_forecast_mat =  np.full((numOfSeries, output_size), 0.0)

    for net in range(NUM_OF_EXPERTS):
        if(len(listOfAssignmentIndices[net])>0):
            print("Testing net "+str(net))
            tester = ExpertTester(**model_kwargs)  # CHANGE THIS IF POSSIBLE AND INITIALIZE THIS AS AN ARRAY
            (listOfAssignmentIndices[net]).sort()

            filter_data(files['validation_file_path'], files['filtered_validation_file_path'], listOfAssignmentIndices[net])
            # filter_data(files['test_file_path'], files['filtered_test_file_path'], listOfAssignmentIndices[net])

            writeTfRecords('', files['filtered_validation_file_path'], '', '', files['filtered_binary_validation_file_path'], '')

            tfrecord_reader = TFRecordReader(model_kwargs['input_size'], model_kwargs['output_size'], meta_data_size)

            training_dataset = tf.data.TFRecordDataset(filenames=[files['filtered_binary_validation_file_path']], compression_type="ZLIB")
            training_dataset = training_dataset.repeat(count=int(model_kwargs['max_epoch_size']))
            training_dataset = training_dataset.map(tfrecord_reader.validation_data_parser)

            test_dataset = tf.data.TFRecordDataset([files['binary_test_file_path']], compression_type="ZLIB")
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

            # count = 0
            for f in range(numOfSeries):
                forecast_mat[f,net] = forecasts[f]
                # count = count + 1

    for il in range(numOfSeries):
        required_nets = final_rank_mat[il,]
        for f in range(output_size):
           total = 0.0
           for rnet in required_nets:
               total = total + forecast_mat[il,rnet,f]
           # all_forecasts = np.ma.masked_equal(forecast_mat[il, :, f], 0.0)
           # final_forecast_mat[il,f] = np.mean(all_forecasts)
           final_forecast_mat[il, f] = total/TOP_N

    return final_forecast_mat


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Ensemble of Experts")
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
    argument_parser.add_argument('--is_early_stopping', required=False, help='Early stopping or fixed number of iterations. Default is 0')
    argument_parser.add_argument('--num_of_experts', required=True, help='Number od experts to be used')
    argument_parser.add_argument('--hyperparameters', required=True, help='File path containing hyperparameters')
    argument_parser.add_argument('--seed', required=True, help='Integer seed to use as the random seed')

    args = argument_parser.parse_args()

    with open(args.hyperparameters, "r") as read_file:
        optimized_params = json.load(read_file)

    NUM_OF_EXPERTS = int(args.num_of_experts)
    dataset_name = args.dataset_name
    seasonality_period = args.seasonality_period
    contain_zero_values = int(args.contain_zero_values)

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
        'filtered_test_file_path': output_dir + "filtered"+ args.input_file_start_name + "test_" + args.forecast_horizon +"i" + args.input_size + ".txt",
        'filtered_binary_test_file_path': output_binary_dir + "filtered"+ args.input_file_start_name + "test_" + args.forecast_horizon +"i" + args.input_size + ".tfrecords",
        'binary_test_file_path': output_binary_dir + args.input_file_start_name + "test_" + args.forecast_horizon + "i" + args.input_size + ".tfrecords",
        'txt_test_file_path' : args.txt_test_file,
        'actual_results_file_path' : args.actual_results_file,
        'original_data_file_path' : args.original_data_file
    }

    if args.without_stl_decomposition:
        without_stl_decomposition = bool(int(args.without_stl_decomposition))
    else:
        without_stl_decomposition = False

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

    if args.address_near_zero_instability:
        address_near_zero_instability = bool(int(args.address_near_zero_instability))
    else:
        address_near_zero_instability = False

    if args.integer_conversion:
        integer_conversion = bool(int(args.integer_conversion))
    else:
        integer_conversion = False
    if args.is_early_stopping:
        is_early_stopping = bool(int(args.is_early_stopping))
    else:
        is_early_stopping = False
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

    model_identifier = dataset_name + "_" + str(NUM_OF_EXPERTS) + " experts " + cell_type + "cell" + "_" + input_format + "_" + stl_decomposition_identifier + "_" + optimizer + "_" + tbptt_identifier + "_" + accumulated_error_identifier + "_" + str(seed)

    numOfSeries = int(args.num_of_series)
    rangeOfSeries = [i for i in range(0,numOfSeries)]  #Start series from 0
    perf_mat = np.full((numOfSeries, NUM_OF_EXPERTS), 1e300)
    listOfBestIndices = NUM_OF_EXPERTS * [[0]]


    model_kwargs = {
        'use_bias': BIAS,
        'use_peepholes': LSTM_USE_PEEPHOLES,
        'boosting': False,
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
    train_padded_shapes = ([], [tf.Dimension(None), input_size], [tf.Dimension(None), output_size])

    validation_padded_shapes = (
        [], [tf.Dimension(None), input_size], [tf.Dimension(None), output_size],
        [tf.Dimension(None), output_size + 1])


    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    validationErrors = np.full((model_kwargs['max_num_epochs'],NUM_OF_EXPERTS), 0.0)
    badNets = []


    for net in range(NUM_OF_EXPERTS):
        # trainers.append(ExpertTrainer(**model_kwargs))
        listOfAssignmentIndices.append(np.random.choice(rangeOfSeries, int(len(rangeOfSeries)*0.5)))


    for net in range(NUM_OF_EXPERTS):
        listOfAssignmentIndices[net] = list(set(listOfAssignmentIndices[net])) #get unique indices

    # print(listOfAssignmentIndices)
    for epoch in range(model_kwargs['max_num_epochs']):
        for net in range(NUM_OF_EXPERTS):
            if ((not is_early_stopping) or (is_early_stopping and (not (net in badNets)))):
                print("Training net "+str(net)+" with epoch "+str(epoch))
                trainer = ExpertTrainer(**model_kwargs)  # CHANGE THIS IF POSSIBLE AND INITIALIZE THIS AS AN ARRAY

                (listOfAssignmentIndices[net]).sort()

                filter_data(files['train_file_path'], files['filtered_train_file_path'], listOfAssignmentIndices[net])

                writeTfRecords(files['filtered_train_file_path'], '', '', files['filtered_binary_train_file_path'], '', '')

                training_dataset = tf.data.TFRecordDataset(filenames=[files['filtered_binary_train_file_path']], compression_type="ZLIB")
                training_dataset = training_dataset.repeat(count=int(model_kwargs['max_epoch_size']))
                training_dataset = training_dataset.map(tfrecord_reader.train_data_parser)

                validation_dataset = tf.data.TFRecordDataset([files['binary_validation_file_path']], compression_type="ZLIB")
                validation_dataset = validation_dataset.map(tfrecord_reader.validation_data_parser)

                # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
                padded_training_data_batches = training_dataset.padded_batch(batch_size=int(model_kwargs['minibatch_size']),
                                                                             padded_shapes=train_padded_shapes)

                padded_validation_dataset = validation_dataset.padded_batch(batch_size=int(model_kwargs['minibatch_size']),
                                                                            padded_shapes=validation_padded_shapes)

                params = {
                    'num_hidden_layers' : int(round(model_kwargs['num_hidden_layers'])),
                    'cell_dimension' : int(round(model_kwargs['cell_dimension'])),
                    'minibatch_size' : int(round(model_kwargs['minibatch_size'])),
                    'max_epoch_size' : int(round(model_kwargs['max_epoch_size'])),
                    'max_num_epochs' : int(round(model_kwargs['max_num_epochs'])),
                    'l2_regularization' : model_kwargs['l2_regularization'],
                    'gaussian_noise_stdev' : model_kwargs['gaussian_noise_stdev'],
                    'random_normal_initializer_stdev' : model_kwargs['random_normal_initializer_stdev'],
                    'optimizer' : optimizer,
                    'padded_training_data_batches' : padded_training_data_batches,
                    'padded_validation_data' : padded_validation_dataset
                }

                p = multiprocessing.Process(target=trainer.test_model,args=(params,return_dict))     # USE TRAINERS[NET]
                p.start()
                p.join()
                smape_errors = return_dict[0]
                print(smape_errors)
                validationErrors[epoch,net] = np.mean(smape_errors)

                if(is_early_stopping):
                    if((epoch==0) or ((epoch>0) and (validationErrors[(epoch-1),net] > validationErrors[epoch,net]))):
                        count = 0
                        for error in smape_errors:
                            perf_mat[count, net] = error
                            count = count + 1
                else:
                    count = 0
                    for error in smape_errors:
                        perf_mat[count, net] = error
                        count = count + 1

                if(is_early_stopping):
                    if((epoch>0) and (validationErrors[(epoch-1),net] < validationErrors[epoch,net])):
                        badNets.append(net)
                        listOfAssignmentIndices[net] = previousIndices[net]


        print(validationErrors)

        if(is_early_stopping):
            previousIndices = copy.deepcopy(listOfAssignmentIndices)

        if(is_early_stopping and (len(badNets) > 0)):
            print("Validation error is growing for "+str(badNets))

        if(is_early_stopping and (len(badNets)==NUM_OF_EXPERTS)):
            break

        rank_mat = np.full((numOfSeries, NUM_OF_EXPERTS), -1)

        for il in range(numOfSeries):
            rank_mat[il,] = np.argsort(perf_mat[il,])

        bestAssignments = rank_mat[:, 0];

        for net in range(NUM_OF_EXPERTS):
            if ((not is_early_stopping) or (is_early_stopping and (not (net in badNets)))):
                listOfBestIndices[net] = np.where(bestAssignments == net)[0]

        for net in range(NUM_OF_EXPERTS):
            if ((not is_early_stopping) or (is_early_stopping and (not (net in badNets)))):
                listOfAssignmentIndices[net] = listOfBestIndices[net]
                (listOfAssignmentIndices[net]).sort()
                for itn in range(1, TOP_N):
                    assignments = rank_mat[:, itn]
                    otherBestIndices = np.where(assignments == net)[0]
                    listOfAssignmentIndices[net] = np.concatenate((listOfAssignmentIndices[net], otherBestIndices))
                    (listOfAssignmentIndices[net]).sort()


                if ((len(listOfBestIndices[net]) == 0) and (epoch!=(model_kwargs['max_num_epochs']-1))) :
                    print("restarting net ", net)
                    listOfAssignmentIndices[net] = np.random.choice(rangeOfSeries, int(len(rangeOfSeries) * 0.5))
                    listOfAssignmentIndices[net] = list(set(listOfAssignmentIndices[net]))
                    (listOfAssignmentIndices[net]).sort()


    print(listOfAssignmentIndices)
    final_rank_mat = np.full((numOfSeries, TOP_N), -1)
    for il in range(numOfSeries):
        rank_mat[il,] = np.argsort(perf_mat[il,])
        for itn in range(TOP_N):
            final_rank_mat[:,itn] = rank_mat[:, itn]

    # print(final_rank_mat)
    final_forecasts = getForecasts(files,model_kwargs,return_dict,final_rank_mat,NUM_OF_EXPERTS)

    rnn_forecasts_full_file_path = BASE_DIR + model_testing_configs.RNN_FORECASTS_DIRECTORY + model_identifier + '.txt'
    validation_error_file_path = BASE_DIR + model_training_configs.VALIDATION_ERRORS_DIRECTORY + model_identifier + '.txt'

    with open(rnn_forecasts_full_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(final_forecasts)

    with open(validation_error_file_path, "w") as output2:
        writer = csv.writer(output2, lineterminator='\n')
        writer.writerows(validationErrors)

    error_file_name = model_identifier + '.txt'
    rnn_forecasts_file_path = model_testing_configs.RNN_FORECASTS_DIRECTORY + model_identifier + '.txt'
    invoke_r_script((rnn_forecasts_file_path, error_file_name, files['txt_test_file_path'],
                     files['actual_results_file_path'], files['original_data_file_path'], str(input_size), str(output_size),
                     str(contain_zero_values), str(int(address_near_zero_instability)),
                     str(int(integer_conversion)), str(int(seasonality_period)), str(int(without_stl_decomposition))),
                    True)

    print("finished")



