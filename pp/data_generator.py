import numpy as np
import keras
import copy


class DataGenerator(keras.utils.Sequence):


    """Generates data for Keras"""
    def __init__(self, _preprocessor, list_ids, cropped_process_instances, cropped_time_deltas, cropped_context_attributes,
                 cropped_outcome_values, next_events, args, batch_size=32, shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.next_events = next_events
        self.data_structure = copy.deepcopy(_preprocessor.data_structure)
        self.preprocessor = _preprocessor
        self.list_ids = list_ids
        self.cropped_process_instances = cropped_process_instances
        self.cropped_time_deltas = cropped_time_deltas
        self.cropped_context_attributes = cropped_context_attributes
        self.cropped_outcome_values = cropped_outcome_values
        self.shuffle = shuffle
        self.on_epoch_end()
        self.args = args

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""

        # select cropped instances for batch
        cropped_process_instances = [self.cropped_process_instances[_id] for _id in list_ids_temp]
        cropped_time_deltas = [self.cropped_time_deltas[_id] for _id in list_ids_temp]
        cropped_outcome_values = [self.cropped_outcome_values[_id] for _id in list_ids_temp]

        if self.cropped_context_attributes:
            cropped_context_attributes = [self.cropped_context_attributes[_id] for _id in list_ids_temp]
        else:
            cropped_context_attributes = self.cropped_context_attributes

        features_data_batch = self.preprocessor.get_data_tensor(self.args,
                                                                cropped_process_instances,
                                                                cropped_time_deltas,
                                                                cropped_context_attributes,
                                                                cropped_outcome_values,
                                                                'train',
                                                                self.data_structure)

        labels_batch = self.preprocessor.get_label_tensor(cropped_process_instances,
                                                          cropped_outcome_values,
                                                          self.data_structure)

        return features_data_batch, labels_batch
