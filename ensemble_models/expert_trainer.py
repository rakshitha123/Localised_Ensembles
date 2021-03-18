# Use with RNN ensemble of specialists model

import numpy as np
import tensorflow as tf
from configs.global_configs import gpu_configs
from external_packages import cocob_optimizer


class ExpertTrainer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
        self.__output_size = kwargs["output_size"]
        self.__seed = kwargs["seed"]
        self.__cell_type = kwargs["cell_type"]

        if kwargs["contain_zero_values"]:
            self.__contain_zero_values = bool(int(kwargs["contain_zero_values"]))
        else:
            self.__contain_zero_values = False

        if kwargs["integer_conversion"]:
            self.__integer_conversion = bool(int(kwargs["integer_conversion"]))
        else:
            self.__integer_conversion = False

        if kwargs["without_stl_decomposition"]:
            self.__without_stl_decomposition = bool(int(kwargs["without_stl_decomposition"]))
        else:
            self.__without_stl_decomposition = False

        if(kwargs["address_near_zero_instability"]):
            self.__address_near_zero_instability = bool(int(kwargs["address_near_zero_instability"]))
        else:
            self.__address_near_zero_instability = False

        if (kwargs["boosting"]):
            self.__boosting = bool(int(kwargs["boosting"]))
        else:
            self.__boosting = False

        self.__num_hidden_layers = kwargs['num_hidden_layers']
        self.__cell_dimension = kwargs['cell_dimension']
        self.__l2_regularization = kwargs['l2_regularization']
        self.__gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        self.__random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']
        self.__learning_rate = kwargs['learning_rate']

        def cell():  # RNN with the layer of cells
            if self.__cell_type == "LSTM":
                cell = tf.nn.rnn_cell.LSTMCell(num_units=int(self.__cell_dimension), use_peepholes=self.__use_peepholes,
                                               initializer=self.__weight_initializer)
            elif self.__cell_type == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(num_units=int(self.__cell_dimension),
                                              kernel_initializer=self.__weight_initializer)
            elif self.__cell_type == "RNN":
                cell = tf.nn.rnn_cell.BasicRNNCell(num_units=int(self.__cell_dimension))
            return cell

        def adagrad_optimizer_fn():
            return tf.train.AdagradOptimizer(learning_rate=self.__learning_rate).minimize(self.__total_loss)

        def adam_optimizer_fn():
            return tf.train.AdamOptimizer(learning_rate=self.__learning_rate).minimize(self.__total_loss)

        def cocob_optimizer_fn():
            return cocob_optimizer.COCOB().minimize(loss=self.__total_loss)

        tf.reset_default_graph()  # reset the tensorflow graph
        tf.set_random_seed(self.__seed)

        # g = "graph"+str(kwargs['net'])
        # globals()[g] = tf.get_default_graph()
        # with globals()[g].as_default():

        self.__input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__input_size])  # declare the input and output placeholders
        self.__noise = tf.random_normal(shape=tf.shape(self.__input), mean=0.0, stddev=self.__gaussian_noise_stdev, dtype=tf.float32)
        self.__training_input = self.__input + self.__noise
        self.__validation_input = self.__input

        self.__true_output = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])  # output format [batch_size, sequence_length, dimension]
        self.__sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

        self.__weight_initializer = tf.truncated_normal_initializer(stddev=self.__random_normal_initializer_stdev)

        self.__multi_layered_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell() for _ in range(int(self.__num_hidden_layers))])

        with tf.variable_scope('train_scope') as train_scope:
            self.__training_rnn_outputs, self.__training_rnn_states = tf.nn.dynamic_rnn(cell=self.__multi_layered_cell,
                                                                                        inputs=self.__training_input,
                                                                                        sequence_length=self.__sequence_lengths,
                                                                                        dtype=tf.float32)

            # connect the dense layer to the RNN
            self.__training_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=self.__training_rnn_outputs, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=self.__weight_initializer, name='dense_layer')

        with tf.variable_scope(train_scope, reuse=tf.AUTO_REUSE) as inference_scope:
            self.__inference_rnn_outputs, self.__inference_rnn_states = tf.nn.dynamic_rnn(
                cell=self.__multi_layered_cell,
                inputs=self.__validation_input,
                sequence_length=self.__sequence_lengths,
                dtype=tf.float32)
            # connect the dense layer to the RNN
            self.__inference_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=self.__inference_rnn_outputs, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=self.__weight_initializer, name='dense_layer', reuse=True)

        self.__error = self.__l1_loss(self.__training_prediction_output, self.__true_output)  # error that should be minimized in the training process

        self.__l2_loss = 0.0  # l2 regularization of the trainable model parameters
        for var in tf.trainable_variables():
            self.__l2_loss += tf.nn.l2_loss(var)

        self.__l2_loss = tf.multiply(tf.cast(self.__l2_regularization, dtype=tf.float64),
                                     tf.cast(self.__l2_loss, dtype=tf.float64))
        self.__total_loss = tf.cast(self.__error, dtype=tf.float64) + self.__l2_loss

        if kwargs['optimizer'] == "cocob":
            self.__optimizer = cocob_optimizer_fn()
        elif kwargs['optimizer'] == "adagrad":
            self.__optimizer = adagrad_optimizer_fn()
        elif kwargs['optimizer'] == "adam":
            self.__optimizer = adam_optimizer_fn()

        self.__shuffle_seed = tf.placeholder(dtype=tf.int64, shape=[])  # prepare the training data into batches



    def __l1_loss(self, z, t):
        loss = tf.reduce_mean(tf.abs(t - z))
        return loss


    def test_model(self, kwargs, return_dict):   # Training the time series

        training_data_batch_iterator = kwargs['padded_training_data_batches'].make_initializable_iterator()   # get an iterator to the batches
        next_training_data_batch = training_data_batch_iterator.get_next()   # access each batch using the iterator

        validation_iterator = kwargs['padded_validation_data'].make_initializable_iterator()
        validation_data_batch = validation_iterator.get_next()

        init_op = tf.global_variables_initializer()    # setup variable initialization
        gpu_options = tf.GPUOptions(allow_growth=True)   # define the GPU options


        with tf.Session(
                config=tf.ConfigProto(log_device_placement=gpu_configs.log_device_placement, allow_soft_placement=True,
                                      gpu_options=gpu_options)) as session:
            session.run(init_op)
            smape_list = []
            validation_outputs = []
            errors = []

            for e in range(int(kwargs['max_num_epochs'])):   #Accroding to theory this should be 1 epoch
                print("Epoch->", e)

                session.run(training_data_batch_iterator.initializer, feed_dict={self.__shuffle_seed: e})
                while True:
                    try:
                        training_data_batch_value = session.run(next_training_data_batch,
                                                                feed_dict={self.__shuffle_seed: e})

                        _, total_loss_value = session.run([self.__optimizer, self.__total_loss],
                                                          feed_dict={self.__training_input: training_data_batch_value[1],
                                                                     self.__true_output: training_data_batch_value[2],
                                                                     self.__sequence_lengths: training_data_batch_value[0]})

                    except tf.errors.OutOfRangeError:
                        break

            session.run(validation_iterator.initializer)

            while True:
                try:

                    validation_batch_value = session.run(validation_data_batch)

                    validation_output = session.run(self.__inference_prediction_output,
                                              feed_dict={self.__input: validation_batch_value[1],
                                                         self.__sequence_lengths: validation_batch_value[0]})

                    last_output_index = validation_batch_value[0] - 1
                    array_first_dimension = np.array(range(0, validation_batch_value[0].shape[0]))

                    true_seasonality_values = validation_batch_value[3][array_first_dimension,
                                              last_output_index, 1:]

                    level_values = validation_batch_value[3][array_first_dimension, last_output_index, 0]

                    last_validation_outputs = validation_output[array_first_dimension, last_output_index]
                    actual_values = validation_batch_value[2][array_first_dimension, last_output_index, :]

                    validation_outputs.extend(last_validation_outputs)

                    if self.__boosting:
                        error_values = actual_values - last_validation_outputs
                        errors.extend(error_values)

                    if self.__without_stl_decomposition:
                        converted_validation_output = np.exp(last_validation_outputs)
                        converted_actual_values = np.exp(actual_values)

                    else:
                        converted_validation_output = np.exp(
                            true_seasonality_values + level_values[:, np.newaxis] + last_validation_outputs)
                        converted_actual_values = np.exp(
                            true_seasonality_values + level_values[:, np.newaxis] + actual_values)

                    if self.__contain_zero_values:  # to compensate for 0 values in data
                        converted_validation_output = converted_validation_output - 1
                        converted_actual_values = converted_actual_values - 1

                    if self.__without_stl_decomposition:
                        converted_validation_output = converted_validation_output * level_values[:, np.newaxis]
                        converted_actual_values = converted_actual_values * level_values[:, np.newaxis]

                    if self.__integer_conversion:
                        converted_validation_output = np.round(converted_validation_output)
                        converted_actual_values = np.round(converted_actual_values)

                    converted_validation_output[converted_validation_output < 0] = 0
                    converted_actual_values[converted_actual_values < 0] = 0

                    if self.__address_near_zero_instability:
                        # calculate the smape
                        epsilon = 0.1
                        sum = np.maximum(
                            np.abs(converted_validation_output) + np.abs(converted_actual_values) + epsilon,
                            0.5 + epsilon)
                        smape_values = (np.abs(converted_validation_output - converted_actual_values) /
                                        sum) * 2
                        smape_values_per_series = np.mean(smape_values, axis=1)
                        smape_list.extend(smape_values_per_series)
                    else:
                        # calculate the smape
                        smape_values = (np.abs(converted_validation_output - converted_actual_values) /
                                        (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
                        smape_values_per_series = np.mean(smape_values, axis=1)
                        smape_list.extend(smape_values_per_series)

                    # if self.__boosting:
                    #     error_values = converted_actual_values - converted_validation_output
                    #     errors.extend(error_values)

                except tf.errors.OutOfRangeError:
                    break

            session.close()

        return_dict[0] = smape_list
        return_dict[4] = validation_outputs
        return_dict[5] = errors
        # return smape_list
