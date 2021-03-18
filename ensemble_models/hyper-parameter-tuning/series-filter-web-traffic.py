# Randomly choose 1/7 series for hyperparameter tuning

import random
import os
from tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter


BASE_DIR = "Localised_Ensembles"
output_dir = BASE_DIR + 'datasets/text_data/kaggle_web_traffic/moving_window/hyper-parameter_tuning/'
output_binary_dir = BASE_DIR + 'datasets/binary_data/kaggle_web_traffic/moving_window/hyper-parameter_tuning/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_binary_dir):
    os.makedirs(output_binary_dir)

args = {
    'num_of_trainers': 7,
    'input_size': 9,
    'output_size': 59,
    'num_of_series': 997,
    'file_path': BASE_DIR + 'datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt',
    'train_file_path': BASE_DIR + 'datasets/text_data/kaggle_web_traffic/moving_window/kaggle_stl_59i9.txt',
    'validation_file_path': BASE_DIR + 'datasets/text_data/kaggle_web_traffic/moving_window/kaggle_stl_59i9v.txt',
    'filtered_train_file_path': output_dir + 'filtered_kaggle_stl_59i9.txt',
    'filtered_validation_file_path': output_dir + 'filtered_kaggle_stl_59i9v.txt',
    'filtered_binary_train_file_path': output_binary_dir + 'filtered_kaggle_stl_59i9.tfrecords',
    'filtered_binary_validation_file_path': output_binary_dir + 'filtered_kaggle_stl_59i9v.tfrecords'
}



def filter_data(original_file_path, filtered_file_path, assignedIndexes):
    original_file = open(original_file_path, "r")
    filtered_file = open(filtered_file_path, "w")

    for line in original_file:
        currentSeriesIndex = int((line.split("|"))[0])
        if (currentSeriesIndex in assignedIndexes):
            filtered_file.write(line)

    original_file.close()
    filtered_file.close()


def writeTfRecords(trainPath, validationPath, binaryTrainPath, binaryValidationPath):
    tfrecord_writer = TFRecordWriter(
        input_size = args['input_size'],
        output_size = args['output_size'],
        train_file_path = trainPath,
        validate_file_path = validationPath,
        test_file_path = '',
        binary_train_file_path = binaryTrainPath,
        binary_validation_file_path = binaryValidationPath,
        binary_test_file_path = ''
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()



if __name__ == '__main__':

    numOfSeries = args['num_of_series']
    rangeOfSeries = [i for i in range(0, numOfSeries)]
    chosenIndices = random.sample(rangeOfSeries, int(len(rangeOfSeries) * (1/args['num_of_trainers'])))

    filter_data(args['train_file_path'], args['filtered_train_file_path'], chosenIndices)
    filter_data(args['validation_file_path'], args['filtered_validation_file_path'], chosenIndices)

    writeTfRecords(args['filtered_train_file_path'], args['filtered_validation_file_path'], args['filtered_binary_train_file_path'], args['filtered_binary_validation_file_path'])
