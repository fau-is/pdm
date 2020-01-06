from __future__ import print_function, division
import keras
from datetime import datetime


def train(args, preprocessor):
    preprocessor.set_training_set()

    features_data = preprocessor.data_structure['data']['train']['features_data']
    labels = preprocessor.data_structure['data']['train']['labels']
    max_length_process_instance = preprocessor.data_structure['meta']['max_length_process_instance']
    num_features = preprocessor.data_structure['meta']['num_features']
    num_classes = preprocessor.data_structure['meta']['num_classes']

    print('Create machine learning model ... \n')

    if args.dnn_architecture == 0:

        # input layer
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # hidden layer
        b1 = keras.layers.Bidirectional(
            keras.layers.recurrent.LSTM(100, use_bias=True, implementation=1, activation="tanh", kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2))(main_input)
        # b1 = keras.layers.normalization.BatchNormalization()(l1)

        # output layer
        act_output = keras.layers.core.Dense(num_classes, activation='softmax', name='act_output', kernel_initializer='glorot_uniform')(b1)

    model = keras.models.Model(inputs=[main_input], outputs=[act_output])

    optimizer = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                       schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_%s.h5' % (args.checkpoint_dir,
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

    # for layer in model.layers:
    #    print(layer.get_config(), layer.get_weights())

    """
    Assumptions:
        - bias bxh_left and bxh_right is not stored by keras
        - bias of output layer is also set to 0
        - remark: keras does not output 2 weight vectors for why -> we use lstm instead of bi-lstm
    """

    # test
    # kernel left lstm layer
    wxh_left = model.layers[1].get_weights()[0]
    # recurrent kernel left lstm layer
    whh_left = model.layers[1].get_weights()[1]
    # biases left lstm layer
    bhh_left = model.layers[1].get_weights()[2]

    # kernel right lstm layer
    wxh_right = model.layers[1].get_weights()[3]
    # recurrent kernel right lstm layer
    whh_right = model.layers[1].get_weights()[4]
    # biases right lstm layer
    bhh_right = model.layers[1].get_weights()[5]

    # linear output layer
    why_left = model.layers[1].get_weights()[0]
    why_right = model.layers[1].get_weights()[0]

    print(wxh_left)  # 4d*e // e = input neurons; d = hidden neurons
    print(whh_left)  # 4d*4
    print(bhh_left)  # 4d

    print(wxh_right)
    print(whh_right)
    print(bhh_right)

    print(why_left)  # C*d // C = classes
    print(why_right)  # C*d

    print(0)

    return training_time.total_seconds()
