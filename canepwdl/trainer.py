from __future__ import print_function, division
import keras_contrib
import keras
from datetime import datetime
from canepwdl.data_generator import DataGenerator
import canepwdl.utils as utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import tensorflow as tf

def train(args, preprocessor):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    cropped_process_instances, cropped_time_deltas, cropped_context_attributes, cropped_outcome_values, next_events =\
        preprocessor.get_training_set()

    max_length_process_instance = preprocessor.data_structure['meta']['max_length_process_instance']
    num_features = preprocessor.data_structure['encoding']['num_values_features'] - 1  # case
    if not args.include_time_delta:
        num_features -= 1
    num_outcome_values = len(preprocessor.data_structure['support']['outcome_types'])

    data_ids, validation_ids, _, _ = utils.split_train_test(preprocessor.data_structure['data']['train']['ids_process_instances'])

    training_generator = DataGenerator(
        preprocessor,
        data_ids,
        cropped_process_instances,
        cropped_time_deltas,
        cropped_context_attributes,
        cropped_outcome_values,
        next_events,
        args,
        args.batch_size_train)

    validation_generator = DataGenerator(
        preprocessor,
        validation_ids,
        cropped_process_instances,
        cropped_time_deltas,
        cropped_context_attributes,
        cropped_outcome_values,
        next_events,
        args,
        args.batch_size_train)


    print('Create deep learning model ... \n')

    if args.dnn_architecture == 0:
        """
        Multi-layer perceptron (MLP)
        Note:
            - flatten/reshape because when multivariate all should be on the same axis
        Configuration according to the work "Decay Replay Mining to Predict Next Process Events" from Theis et al. (2019).
            - optimizer = adam
            - loss = categorical crossentropy
            - learning-rate = 0.001
        """

        # layer 1
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        input_layer_flattened = keras.layers.Flatten()(main_input)

        # layer 2
        hidden_layer_1 = keras.layers.Dense(300, activation='relu')(input_layer_flattened)
        hidden_layer_1 = keras.layers.normalization.BatchNormalization()(hidden_layer_1)
        hidden_layer_1 = keras.layers.Dropout(0.5)(hidden_layer_1)

        # layer 3
        hidden_layer_2 = keras.layers.Dense(200, activation='relu')(hidden_layer_1)
        hidden_layer_2 = keras.layers.normalization.BatchNormalization()(hidden_layer_2)
        hidden_layer_2 = keras.layers.Dropout(0.5)(hidden_layer_2)

        # layer 4
        hidden_layer_3 = keras.layers.Dense(100, activation='relu')(hidden_layer_2)
        hidden_layer_3 = keras.layers.normalization.BatchNormalization()(hidden_layer_3)
        hidden_layer_3 = keras.layers.Dropout(0.5)(hidden_layer_3)

        # layer 5
        hidden_layer_4 = keras.layers.Dense(50, activation='relu')(hidden_layer_3)
        hidden_layer_4 = keras.layers.normalization.BatchNormalization()(hidden_layer_4)
        hidden_layer_output = keras.layers.Dropout(0.5)(hidden_layer_4)

        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    if args.dnn_architecture == 1:
        """
        Long Short-Term Memory Neural Network (LSTM)
        Note:
            - tax et al. do not set a activation function -> if no activation function set -> no activation function is applied -> (f(x)=x)
            - relu activation leads to error if sequences getting longer
            - the original architecture consists of two channels; one for events and the other one for time; we use only the event channel
        Configuration according to the work "Predictive Business Process Monitoring with LSTM Neural Networks" from Tax et al. (2017) 
            - optimizer = nadam
            - loss = categorical crossentropy
            - learning rate = 0.002
        """

        # layer 1
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # layer 2
        hidden_layer_1 = keras.layers.recurrent.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(main_input)
        hidden_layer_output = keras.layers.normalization.BatchNormalization()(hidden_layer_1)

        # layer 3
        # hidden_layer_2 = keras.layers.recurrent.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(hidden_layer_1)
        # hidden_layer_output = keras.layers.normalization.BatchNormalization()(hidden_layer_2)

        optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004, clipvalue=3)


    elif args.dnn_architecture == 2:
        """
        Convolutional neural network (CNN)
        Configuration according to the work "Predicting the Next Process Event Using Convolutional Neural Networks" from Abdulrhman et al. (2019)
            - optimizer = adam
            - loss = categorical crossentropy
            - learning-rate = 0.001
        """

        # layer 1
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # layer 2
        hidden_layer_1 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(main_input)
        hidden_layer_1 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_1)

        # layer 3
        hidden_layer_2 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(hidden_layer_1)
        hidden_layer_2 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_2)

        # layer 4
        hidden_layer_3 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(hidden_layer_2)
        hidden_layer_3 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_3)

        # layer 5
        hidden_layer_4 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(hidden_layer_3)
        hidden_layer_4 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_4)

        # layer 6
        hidden_layer_5 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(hidden_layer_4)
        hidden_layer_5 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_5)

        # layer 7
        hidden_layer_6 = keras.layers.Flatten()(hidden_layer_5)

        # layer 8
        hidden_layer_output = keras.layers.Dense(100, activation='relu')(hidden_layer_6)

        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    elif args.dnn_architecture == 3:
        """
        Encoder
        Configuration according to the work "Deep learning for time series classification: a review" from Fawaz et al. (2019); 
        Originally proposed in the work "Towards a Universal Neural Network Encoder for Time Series" by Serra et al. (2018).
            - optimizer = adam
            - loss = categorical crossentropy
            - learning-rate = 0.00001
        """

        # layer 1
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # layer 2
        hidden_layer_1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(main_input)
        hidden_layer_1 = keras_contrib.layers.InstanceNormalization()(hidden_layer_1)
        hidden_layer_1 = keras.layers.PReLU(shared_axes=[1])(hidden_layer_1)
        hidden_layer_1 = keras.layers.core.Dropout(rate=0.2)(hidden_layer_1)
        hidden_layer_1 = keras.layers.MaxPooling1D(pool_size=2)(hidden_layer_1)

        # layer 3
        hidden_layer_2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(hidden_layer_1)
        hidden_layer_2 = keras_contrib.layers.InstanceNormalization()(hidden_layer_2)
        hidden_layer_2 = keras.layers.PReLU(shared_axes=[1])(hidden_layer_2)
        hidden_layer_2 = keras.layers.core.Dropout(rate=0.2)(hidden_layer_2)
        hidden_layer_2 = keras.layers.MaxPooling1D(pool_size=2)(hidden_layer_2)

        # layer 4
        hidden_layer_3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(hidden_layer_2)
        hidden_layer_3 = keras_contrib.layers.InstanceNormalization()(hidden_layer_3)
        hidden_layer_3 = keras.layers.PReLU(shared_axes=[1])(hidden_layer_3)
        hidden_layer_3 = keras.layers.core.Dropout(rate=0.2)(hidden_layer_3)

        # layer 5
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(hidden_layer_3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(hidden_layer_3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])

        # layer 6
        dense_layer = keras.layers.core.Dense(units=256, activation='sigmoid')(multiply_layer)
        dense_layer = keras_contrib.layers.InstanceNormalization()(dense_layer)

        # layer 7
        hidden_layer_output = keras.layers.Flatten()(dense_layer)

        optimizer = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)


    event_output = keras.layers.core.Dense(num_outcome_values,
                                           activation='softmax',
                                           name='event_output',
                                           kernel_initializer='glorot_uniform')(hidden_layer_output)
    model = keras.models.Model(inputs=[main_input], outputs=[event_output])


    model.compile(loss={'event_output': 'categorical_crossentropy'}, optimizer=optimizer, metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_%s.h5' % (args.checkpoint_dir,
                                                                          preprocessor.data_structure['support'][
                                                                              'iteration_cross_validation']),
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.5,
                                                   patience=10,
                                                   verbose=0,
                                                   mode='auto',
                                                   min_delta=0.0001,
                                                   cooldown=0,
                                                   min_lr=0)
    model.summary()
    start_training_time = datetime.now()

    history = model.fit_generator(
              generator=training_generator,
              validation_data=validation_generator,
              workers=1,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    # print(history.history.keys())

    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model Accuracy and Loss')
    pyplot.ylabel('Loss/Accuracy')
    pyplot.xlabel('# Epoch')
    pyplot.legend(['Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss'], loc='upper left')
    pyplot.savefig(str(args.result_dir + 'model.pdf'))
    pyplot.close()

    return training_time.total_seconds()