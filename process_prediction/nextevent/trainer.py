from __future__ import print_function, division
import keras
from datetime import datetime


def train(args, preprocessor):
    preprocessor.set_training_set()

    features_data = preprocessor.data_structure['data']['train']['features_data']
    labels = preprocessor.data_structure['data']['train']['labels']
    max_length_process_instance = preprocessor.data_structure['meta']['max_length_process_instance']
    num_features = preprocessor.data_structure['meta']['num_features']
    num_event_ids = preprocessor.data_structure['meta']['num_event_ids']

    print('Create machine learning model ... \n')

    if args.dnn_architecture == 0:
        # input layer
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # hidden layer
        b1 = keras.layers.Bidirectional(
            keras.layers.recurrent.LSTM(100, use_bias=True, implementation=1, activation="tanh",
                                        kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2))(
            main_input)

        # output layer
        act_output = keras.layers.core.Dense(num_event_ids, activation='softmax', name='act_output',
                                             kernel_initializer='glorot_uniform')(b1)

    model = keras.models.Model(inputs=[main_input], outputs=[act_output])

    optimizer = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                       schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint('%s%smodel_%s.h5' % (args.task, args.model_dir[1:],
                                                                            preprocessor.data_structure['support'][
                                                                                'iteration_cross_validation']),
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()

    start_training_time = datetime.now()

    model.fit(features_data, {'act_output': labels}, validation_split=1 / args.num_folds, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    return training_time.total_seconds()
