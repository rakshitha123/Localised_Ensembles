# Use with RNN ensemble of specialists model

import numpy as np
import tensorflow as tf
from configs.global_configs import gpu_configs
from external_packages import cocob_optimizer


class ExpertTester:

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

        self.__num_hidden_layers = kwargs['num_hidden_layers']
        self.__cell_dimension = kwargs['cell_dimension']
        self.__l2_regularization = kwargs['l2_regularization']
        self.__gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        self.__random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']
        self.__learning_rate = kwargs['learning_rate']

        def cell():  # RNN with the layer of cells
            if self.__cell_type == "LSTM":
                cell = tf.nn.rnn_cell.LSTMCell(num_units=int(self.__cell_dimension), use_peepholes=self.__use_peepholes, initializer=self.__weight_initializer)
            elif self.__cell_type == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(num_units=int(self.__cell_dimension), kernel_initializer=self.__weight_initializer)
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

        self.__input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__input_size])  # declare the input and output placeholders
        self.__noise = tf.random_normal(shape=tf.shape(self.__input), mean=0.0, stddev=self.__gaussian_noise_stdev, dtype=tf.float32)
        self.__training_input = self.__input + self.__noise
        self.__testing_input = self.__input

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
                inputs=self.__testing_input,
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


    def test_model(self, kwargs, return_dict):

        training_data_batch_iterator = kwargs['padded_training_data_batches'].make_initializable_iterator()   # get an iterator to the batches
        next_training_data_batch = training_data_batch_iterator.get_next()   # access each batch using the iterator

        test_input_iterator = kwargs['padded_test_input_data'].make_one_shot_iterator()
        test_input_data_batch = test_input_iterator.get_next()

        init_op = tf.global_variables_initializer()    # setup variable initialization
        gpu_options = tf.GPUOptions(allow_growth=True)   # define the GPU options


        with tf.Session(
                config=tf.ConfigProto(log_device_placement=gpu_configs.log_device_placement, allow_soft_placement=True,
                                      gpu_options=gpu_options)) as session:
            session.run(init_op)

            for epoch in range(int(kwargs['max_num_epochs'])):   #Accroding to theory this should be 1 epoch
                print("Epoch->", epoch, " ", kwargs['max_num_epochs'])

                session.run(training_data_batch_iterator.initializer, feed_dict={self.__shuffle_seed: epoch})
                while True:
                    try:
                        training_data_batch_value = session.run(next_training_data_batch,
                                                                feed_dict={self.__shuffle_seed: epoch})

                        session.run(self.__optimizer,
                                    feed_dict={self.__input: training_data_batch_value[1],
                                               self.__true_output: training_data_batch_value[2],
                                               self.__sequence_lengths: training_data_batch_value[0]})

                    except tf.errors.OutOfRangeError:
                        break

            # applying the model to the test data

            list_of_forecasts = []

            while True:
                try:

                    # get the batch of test inputs
                    test_input_batch_value = session.run(test_input_data_batch)

                    # get the output of the network for the test input data batch
                    test_output = session.run(self.__inference_prediction_output,
                                              feed_dict={self.__input: test_input_batch_value[1],
                                                         self.__sequence_lengths: test_input_batch_value[0]})

                    last_output_index = test_input_batch_value[0] - 1
                    array_first_dimension = np.array(range(0, test_input_batch_value[0].shape[0]))
                    forecasts = test_output[array_first_dimension, last_output_index]
                    list_of_forecasts.extend(forecasts.tolist())

                except tf.errors.OutOfRangeError:
                    break

            session.close()
            return_dict[1] = list_of_forecasts
