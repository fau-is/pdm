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
        Trial ID which identifies the best trial, if hyper-parameter optimization is performed. Otherwise is -1.

    """

    # gpu sharing
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    # gpu sharing
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


    cases_of_fold = preprocessor.get_cases_of_fold(event_log, train_indices)
    subseq_cases_of_fold = preprocessor.get_subsequences_of_cases(cases_of_fold)

    if args.hpo:
        hpo.create_data(args, event_log, preprocessor, cases_of_fold)

        if args.seed:
            sampler = optuna.samplers.TPESampler(
                seed=args.seed_val)  # Make the sampler behave in a deterministic way.
        else:
            sampler = optuna.samplers.TPESampler()

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
        return train_model(args, event_log, preprocessor, subseq_cases_of_fold), -1


def find_best_model(trial):
    """
    Executes hyper-parameter optimization according to specified hyper-parameter search spaces (as configured in config)
    and builds a model whose hyper-parameter configuration produces the best prediction performance.

    Parameters
    ----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating the objective function 'find_best_model'.

    Returns
    -------
    float :
        Value for evaluation parameter to be optimized during hyper-parameter optimization.

    """

    args = hpo.args
    max_case_length = hpo.max_case_length
    num_features = hpo.num_features
    num_classes = hpo.num_classes

    x_train = hpo.x_train
    y_train = hpo.y_train
    x_test = hpo.x_test
    y_test = hpo.y_test

    print('Create machine learning model ... \n')

    """
    Bi-directional long short-term neural network
    """

    # input layer
    main_input = tf.keras.layers.Input(shape=(max_case_length, num_features), name='main_input')

    # hidden layer
    b1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(trial.suggest_categorical("number_of_units", args.hpo_units),
                             activation=trial.suggest_categorical("activation", args.hpo_activation),
                             kernel_initializer=trial.suggest_categorical("kernel_initializer",
                                                                          args.hpo_kernel_initializer),
                             return_sequences=False,
                             dropout=trial.suggest_discrete_uniform('drop_out_1', 0.1, 0.5, 0.1)))(main_input)

    # output layer
    out_output = tf.keras.layers.Dense(num_classes,
                                       activation='softmax',
                                       name='out_output',
                                       kernel_initializer=trial.suggest_categorical("kernel_initializer",
                                                                                    args.hpo_kernel_initializer)
                                       )(b1)


    model = tf.keras.models.Model(inputs=[main_input], outputs=[out_output])
    model.compile(loss={'out_output': 'categorical_crossentropy'},
                  optimizer=trial.suggest_categorical("optimizer", args.hpo_optimizer),
                  metrics=['accuracy', utils.f1_score])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%s%smodel_%s.h5' % (args.task, args.model_dir[1:],
                                                                                  args.data_set[:-4]),
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()
    model.fit(x_train, y_train, validation_split=args.val_split, verbose=1,
                        callbacks=[early_stopping, model_checkpoint, lr_reducer],
                        batch_size=args.batch_size_train,
                        epochs=args.dnn_num_epochs,
                        class_weight={0: 1, 1: 1}
                        )

    score = model.evaluate(x_test, y_test, verbose=0)
    # We optimize parameters in terms of Accuracy
    return score[1]


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


    """
    Bi-directional long short-term neural network
    """

    # input layer
    main_input = tf.keras.layers.Input(shape=(max_case_length, num_features), name='main_input')

    # hidden layer
    b1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, use_bias=True, implementation=1, activation="tanh",
                             kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2))(main_input)

    # output layer
    out_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='out_output',
                                       kernel_initializer='glorot_uniform')(b1)

    optimizer = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                          schedule_decay=0.004, clipvalue=3)

    model = tf.keras.models.Model(inputs=[main_input], outputs=[out_output])
    model.compile(loss={'out_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%s%smodel.h5' % (args.task, args.model_dir[1:]),
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
