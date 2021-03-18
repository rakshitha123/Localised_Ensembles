# configs for the RNN model training
class model_training_configs:
    VALIDATION_ERRORS_DIRECTORY = 'Localised_Ensembles/results/validation_errors/'
    INFO_FREQ = 1

# configs for the model testing
class model_testing_configs:
    RNN_FORECASTS_DIRECTORY = 'Localised_Ensembles/results/rnn_forecasts/' #forecasts before rescaling
    RNN_ERRORS_DIRECTORY = 'Localised_Ensembles/results/errors'
    PROCESSED_RNN_FORECASTS_DIRECTORY = 'Localised_Ensembles/results/forecasts/'

# configs for hyperparameter tuning(SMAC3)
class hyperparameter_tuning_configs:
    SMAC_RUNCOUNT_LIMIT = 50

class training_data_configs:
    SHUFFLE_BUFFER_SIZE = 20000

class gpu_configs:
    log_device_placement = False
