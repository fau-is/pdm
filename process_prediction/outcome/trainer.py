from __future__ import print_function, division
from datetime import datetime
import tensorflow as tf


def train(args, preprocessor):

    preprocessor.set_training_set(args)

    features_data = preprocessor.data_structure['data']['train']['features_data']
    labels = preprocessor.data_structure['data']['train']['labels']
    max_length_process_instance = preprocessor.data_structure['meta']['max_length_process_instance']
    num_features = preprocessor.data_structure['meta']['num_features']
    num_classes = preprocessor.data_structure['meta']['num_classes']

    print('Create machine learning model ... \n')

    if args.dnn_architecture == 0:
        """
        Bi-directional long short-term neural network
        """

        # input layer
        main_input = tf.keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # hidden layer
        b1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100, use_bias=True, implementation=1, activation="tanh",
                                 kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2))(
            main_input)

        # output layer
        out_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='out_output',
                                           kernel_initializer='glorot_uniform')(b1)

        optimizer = tf.keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                              schedule_decay=0.004, clipvalue=3)

    elif args.dnn_architecture == 1:
        """
        Multi-layer perceptron
        """

        # input layer
        main_input = tf.keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        input_layer_flattened = tf.keras.layers.Flatten()(main_input)

        # layer 2
        hidden_layer_1 = tf.keras.layers.Dense(100, activation='relu')(input_layer_flattened)
        hidden_layer_1 = tf.keras.layers.BatchNormalization()(hidden_layer_1)
        hidden_layer_1 = tf.keras.layers.Dropout(0.5)(hidden_layer_1)

        # layer 3
        hidden_layer_2 = tf.keras.layers.Dense(200, activation='relu')(hidden_layer_1)
        hidden_layer_2 = tf.keras.layers.BatchNormalization()(hidden_layer_2)
        hidden_layer_2 = tf.keras.layers.Dropout(0.5)(hidden_layer_2)

        # layer 4
        hidden_layer_3 = tf.keras.layers.Dense(100, activation='relu')(hidden_layer_2)
        hidden_layer_3 = tf.keras.layers.BatchNormalization()(hidden_layer_3)
        hidden_layer_3 = tf.keras.layers.Dropout(0.5)(hidden_layer_3)

        # layer 5
        hidden_layer_4 = tf.keras.layers.Dense(50, activation='relu')(hidden_layer_3)
        hidden_layer_4 = tf.keras.layers.BatchNormalization()(hidden_layer_4)
        hidden_layer_4 = tf.keras.layers.Dropout(0.5)(hidden_layer_4)

        # output layer
        out_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='out_output',
                                           kernel_initializer='glorot_uniform')(hidden_layer_4)

        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model = tf.keras.models.Model(inputs=[main_input], outputs=[out_output])
    model.compile(loss={'out_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%s%smodel_%s.h5' % (args.task, args.model_dir[1:],
                                                                               preprocessor.data_structure['support'][
                                                                                   'iteration_cross_validation']),
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()

    start_training_time = datetime.now()

    model.fit(features_data, {'out_output': labels}, validation_split=args.val_split, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    return training_time.total_seconds()
