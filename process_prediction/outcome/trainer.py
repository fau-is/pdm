from __future__ import print_function, division
from datetime import datetime
import tensorflow as tf
import optuna
import process_prediction.outcome.hyperparameter_optimization as hpo
import process_prediction.utils as utils


def train(args, event_log, preprocessor, train_indices):
    # TODO description

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

        return training_time.total_seconds()
    else:
        return train_model(args, event_log, preprocessor, cases_of_fold, subseq_cases_of_fold)

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
        hidden_layer_1 = tf.keras.layers.Dense(100, activation=trial.suggest_categorical("activation", args.hpo_activation))(input_layer_flattened)
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
    model.compile(loss={'out_output': 'categorical_crossentropy'}, optimizer=optimizer, metrics=['accuracy',
                                                                                                 utils.f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%s%smodel_%s.h5' % (args.task, args.model_dir[1:],
                                                                               iteration_cross_validation),
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

def train_model(args, event_log, preprocessor, cases_of_fold, subseq_cases_of_fold):
    # TODO Trains model without use of hyperparameter optimization

    features = preprocessor.get_features_tensor(args, 'train', event_log, subseq_cases_of_fold)
    labels = preprocessor.get_labels_tensor(args, cases_of_fold)

    max_case_length = preprocessor.get_max_case_length(event_log)
    num_features = preprocessor.get_num_features(args)
    num_classes = preprocessor.get_num_classes()

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
