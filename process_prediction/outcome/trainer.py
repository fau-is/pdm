from __future__ import print_function, division
from datetime import datetime
import tensorflow as tf
import optuna
import process_prediction.outcome.hyperparameter_optimization as hpo
import process_prediction.utils as utils


def train(args, event_log, preprocessor, train_indices):
    """
    Trains a model for outcome prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    event_log : list of dicts, where single dict represents a case
        pm4py.objects.log.log.EventLog object representing an event log.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    train_indices : list of arrays consisting of ints
        Indices of training cases from event log per fold.

    Returns
    -------
    timedelta :
        Time passed while training a model.
    int :
        Trial ID which identifies the best trial, if hyperparameter optimization is performed. Otherwise is -1.

    """

    cases_of_fold = preprocessor.get_cases_of_fold(event_log, train_indices)
    subseq_cases_of_fold = preprocessor.get_subsequences_of_cases(cases_of_fold)

    if args.hpo:
        hpo.create_data(args, event_log, preprocessor, cases_of_fold)

        sampler = optuna.samplers.TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        start_training_time = datetime.now()
        study.optimize(find_best_model, n_trials=args.hpo_eval_runs)
        training_time = datetime.now() - start_training_time

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return training_time.total_seconds(), study.best_trial.number
    else:
        return train_model(args, event_log, preprocessor, subseq_cases_of_fold), -1

def find_best_model(trial):
    """
    Executes hyperparameter optimization according to specified hyperparameter search spaces (as configured in config)
    and builds a model whose hyperparameter configuration produces the best prediction performance.

    Parameters
    ----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating the objective function 'find_best_model'.

    Returns
    -------
    float :
        Value for evaluation parameter to be optimized during hyperparameter optimization.

    """

    args = hpo.args
    max_case_length = hpo.max_case_length
    num_features = hpo.num_features
    num_classes = hpo.num_classes
    iteration_cross_validation = hpo.iteration_cross_validation

    x_train = hpo.x_train
    y_train = hpo.y_train
    x_test = hpo.x_test
    y_test = hpo.y_test

    print('Create machine learning model ... \n')

    if args.dnn_architecture == 0:
        """
        Bi-directional long short-term neural network
        """

        # input layer
        main_input = tf.keras.layers.Input(shape=(max_case_length, num_features), name='main_input')

        # hidden layer
        b1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(trial.suggest_categorical("number_of_neurons", args.hpo_LSTM),
                                     use_bias=True, implementation=1,
                                     activation=trial.suggest_categorical("activation", args.hpo_activation),
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
        main_input = tf.keras.layers.Input(shape=(max_case_length, num_features), name='main_input')
        input_layer_flattened = tf.keras.layers.Flatten()(main_input)

        # layer 2
        hidden_layer_1 = tf.keras.layers.Dense(trial.suggest_categorical("number_of_neurons", args.hpo_LSTM),
                                               activation=trial.suggest_categorical("activation", args.hpo_activation)
                                               )(input_layer_flattened)
        hidden_layer_1 = tf.keras.layers.BatchNormalization()(hidden_layer_1)
        hidden_layer_1 = tf.keras.layers.Dropout(0.5)(hidden_layer_1)

        # layer 3
        hidden_layer_2 = tf.keras.layers.Dense(trial.suggest_categorical("number_of_neurons", args.hpo_LSTM),
                                               activation=trial.suggest_categorical("activation", args.hpo_activation)
                                               )(hidden_layer_1)
        hidden_layer_2 = tf.keras.layers.BatchNormalization()(hidden_layer_2)
        hidden_layer_2 = tf.keras.layers.Dropout(0.5)(hidden_layer_2)

        # layer 4
        hidden_layer_3 = tf.keras.layers.Dense(trial.suggest_categorical("number_of_neurons", args.hpo_LSTM),
                                               activation=trial.suggest_categorical("activation", args.hpo_activation)
                                               )(hidden_layer_2)
        hidden_layer_3 = tf.keras.layers.BatchNormalization()(hidden_layer_3)
        hidden_layer_3 = tf.keras.layers.Dropout(0.5)(hidden_layer_3)

        # layer 5
        hidden_layer_4 = tf.keras.layers.Dense(trial.suggest_categorical("number_of_neurons", args.hpo_LSTM),
                                               activation=trial.suggest_categorical("activation", args.hpo_activation)
                                               )(hidden_layer_3)
        hidden_layer_4 = tf.keras.layers.BatchNormalization()(hidden_layer_4)
        hidden_layer_4 = tf.keras.layers.Dropout(0.5)(hidden_layer_4)

        # output layer
        out_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='out_output',
                                           kernel_initializer='glorot_uniform')(hidden_layer_4)

        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model = tf.keras.models.Model(inputs=[main_input], outputs=[out_output])
    model.compile(loss={'out_output': 'categorical_crossentropy'}, optimizer=optimizer, metrics=['accuracy',
                                                                                                 utils.f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%s%smodel_%s_trial%s.h5' % (args.task, args.model_dir[1:],
                                                                                       iteration_cross_validation,
                                                                                       trial.number),
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()

    history = model.fit(x_train, {'out_output': y_train}, validation_split=args.val_split, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    print(history.history.keys())

    score = model.evaluate(x_test, y_test, verbose=0)
    return score[2]

def train_model(args, event_log, preprocessor, subseq_cases_of_fold):
    """
    Trains a model for outcome prediction without hyperparameter optimization.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    event_log : list of dicts, where single dict represents a case
        pm4py.objects.log.log.EventLog object representing an event log.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    subseq_cases_of_fold : list of dicts, where single dict represents a subsequence of a case
        Subsequences of the training cases.

    Returns
    -------
    timedelta :
        Time passed while training a model.

    """
    features = preprocessor.get_features_tensor(args, 'train', event_log, subseq_cases_of_fold)
    labels = preprocessor.get_labels_tensor(args, subseq_cases_of_fold)

    max_case_length = preprocessor.get_max_case_length(event_log)
    num_features = preprocessor.get_num_features(args)
    num_classes = preprocessor.get_num_outcome_classes()

    print('Create machine learning model ... \n')

    if args.dnn_architecture == 0:
        """
        Bi-directional long short-term neural network
        """

        # input layer
        main_input = tf.keras.layers.Input(shape=(max_case_length, num_features), name='main_input')

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
        main_input = tf.keras.layers.Input(shape=(max_case_length, num_features), name='main_input')
        input_layer_flattened = tf.keras.layers.Flatten()(main_input)

        # layer 2
        hidden_layer_1 = tf.keras.layers.Dense(300, activation='relu')(input_layer_flattened)
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

    elif args.dnn_architecture == 2:

        # CNN according to Abdulrhman et al. (2019)
        main_input = tf.keras.layers.Input(shape=(max_length_process_instance, num_features), name='input_layer')
        layer_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(
            main_input)
        layer_2 = tf.keras.layers.MaxPool1D()(layer_1)
        layer_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(
            layer_2)
        layer_3 = tf.keras.layers.MaxPool1D()(layer_2)
        layer_3 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(
            layer_3)
        layer_4 = tf.keras.layers.MaxPool1D()(layer_3)
        layer_4 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(
            layer_4)
        layer_4 = tf.keras.layers.MaxPool1D()(layer_4)
        layer_5 = tf.keras.layers.Conv1D(filters=128, kernel_size=16, padding='same', strides=1, activation='relu')(
            layer_4)
        layer_5 = tf.keras.layers.MaxPool1D()(layer_5)
        layer_6 = tf.keras.layers.Flatten()(layer_5)
        layer_7 = tf.keras.layers.Dense(100, activation='relu')(layer_6)

        # output layer
        out_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='out_output',
                                           kernel_initializer='glorot_uniform')(layer_7)

        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model = tf.keras.models.Model(inputs=[main_input], outputs=[out_output])
    model.compile(loss={'out_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%s%smodel_%s.h5' % (args.task, args.model_dir[1:],
                                                                               preprocessor.iteration_cross_validation),
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

    model.fit(features, {'out_output': labels}, validation_split=args.val_split, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    return training_time.total_seconds()
