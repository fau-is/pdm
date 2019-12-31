from __future__ import division
import csv
import numpy
import pandas
import category_encoders
import copy
import canepwdl.utils as utils
from sklearn.model_selection import KFold, ShuffleSplit
import datetime
import gensim


class Preprocessor(object):
    data_structure = {
        'support': {
            'num_folds': 1,
            'data_dir': "",
            'encoded_data_dir': "",
            'ascii_offset': 161,
            'date_format': "%d.%m.%Y %H:%M:%S",  # "%Y-%m-%d %H:%M:%S"
            'train_index_per_fold': [],
            'test_index_per_fold': [],
            'iteration_cross_validation': 0,
            'elements_per_fold': 0,
            'outcome_types': [],
            'map_outcome_type_to_outcome_value': [],
            'map_outcome_value_to_outcome_type': [],
            'doc2vec_model': []
        },

        'encoding': {
            'eventlog_df': pandas.DataFrame,
            'num_values_event_ids': 0,
            'num_values_control_flow': 0,
            'num_values_context_list': [],
            'num_values_context': 0,
            'num_values_outcome': 0,
            'num_values_features': 0,
            'mapping_outcome_values': {},
            'time_deltas': [],
            'is_doc2vec_set': False
        },

        'meta': {
            'max_length_process_instance': 0,
            'num_attributes_context': 0,
            'num_attributes_control_flow': 3,  # process instance id, event id and timestamp
            'num_attributes_outcome': 1,
            'num_process_instances': 0
        },

        'data': {
            'process_instances': [],
            'ids_process_instances': [],
            'context_attributes_process_instances': [],
            'outcome_values': [],

            'train': {
                'process_instances': [],
                'context_attributes': [],
                'outcome_values': [],
                'ids_process_instances': [],
                'time_deltas': [],
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            },

            'test': {
                'process_instances': [],
                'context_attributes': [],
                'outcome_values': [],
                'ids_process_instances': [],
                'time_deltas': []
            }
        }
    }

    def __init__(self, args):
        utils.llprint("Initialization ... \n")

    def create(self, args):
        utils.llprint("Creation ... \n")
        self.data_structure['support']['num_folds'] = args.num_folds
        self.data_structure['support']['data_dir'] = args.data_dir + args.data_set
        self.data_structure['support']['encoded_data_dir'] = r'%s' % args.data_dir + r'encoded_%s' % args.data_set

        encoded_eventlog_df = self.encode_data(args)
        # + 2 -> case + time
        self.data_structure['encoding']['num_values_control_flow'] = \
            2 + self.data_structure['encoding']['num_values_event_ids']
        self.data_structure['encoding']['num_values_outcome'] = \
            self.data_structure['encoding']['num_values_outcome']

        self.get_sequences_from_encoded_eventlog(encoded_eventlog_df)

        self.data_structure['support']['elements_per_fold'] = \
            int(round(self.data_structure['meta']['num_process_instances'] /
                      self.data_structure['support']['num_folds']))

        self.data_structure['meta']['max_length_process_instance'] = max(
            [len(x) for x in self.data_structure['data']['process_instances']])

        self.data_structure['support']['outcome_types'] = list(
            self.data_structure['encoding']['mapping_outcome_values'].values())

        self.data_structure['support']['map_outcome_type_to_outcome_value'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['outcome_types']))
        self.data_structure['support']['map_outcome_value_to_outcome_type'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['outcome_types']))

        if self.data_structure['encoding']['is_doc2vec_set']:
            self.data_structure['encoding']['num_values_features'] = 1 + args.doc2vec_vec_size  # case + event encoding
        else:
            self.data_structure['encoding']['num_values_features'] = self.data_structure['encoding'][
                                                                         'num_values_control_flow'] + \
                                                                     self.data_structure['encoding'][
                                                                         'num_values_context'] + \
                                                                     self.data_structure['encoding'][
                                                                         'num_values_outcome']

        if args.cross_validation:
            self.set_indices_k_fold_validation()
        else:
            self.set_indices_split_validation(args)



    def reset_data_structure(self):
        self.data_structure = {
            'support': {
                'num_folds': 1,
                'data_dir': "",
                'encoded_data_dir': "",
                'ascii_offset': 161,
                'date_format': "%d.%m.%Y %H:%M:%S",  # "%Y-%m-%d %H:%M:%S"
                'train_index_per_fold': [],
                'test_index_per_fold': [],
                'iteration_cross_validation': 0,
                'elements_per_fold': 0,
                'outcome_types': [],
                'map_outcome_type_to_outcome_value': [],
                'map_outcome_value_to_outcome_type': [],
                'doc2vec_model': []
            },

            'encoding': {
                'eventlog_df': pandas.DataFrame,
                'num_values_event_ids': 0,
                'num_values_control_flow': 0,
                'num_values_context_list': [],
                'num_values_context': 0,
                'num_values_outcome': 0,
                'num_values_features': 0,
                'mapping_outcome_values': {},
                'time_deltas': [],
                'is_doc2vec_set': False
            },

            'meta': {
                'max_length_process_instance': 0,
                'num_attributes_context': 0,
                'num_attributes_control_flow': 3,  # process instance id, event id and timestamp
                'num_attributes_outcome': 1,
                'num_process_instances': 0
            },

            'data': {
                'process_instances': [],
                'ids_process_instances': [],
                'context_attributes_process_instances': [],
                'outcome_values': [],

                'train': {
                    'process_instances': [],
                    'context_attributes': [],
                    'outcome_values': [],
                    'ids_process_instances': [],
                    'time_deltas': [],
                    'features_data': numpy.array([]),
                    'labels': numpy.ndarray([])
                },

                'test': {
                    'process_instances': [],
                    'context_attributes': [],
                    'outcome_values': [],
                    'ids_process_instances': [],
                    'time_deltas': []
                }
            }
        }



    def encode_data(self, args):

        utils.llprint("Encoding ... \n")

        eventlog_df = pandas.read_csv(self.data_structure['support']['data_dir'], sep=';')
        self.data_structure['encoding']['eventlog_df'] = eventlog_df

        # check for doc2vec encoding
        if args.encoding_num == 'doc2vec' or args.encoding_cat == 'doc2vec':
            self.data_structure['encoding']['is_doc2vec_set'] = True
            self.build_doc2vec_model(args)

        # add case ID unencoded to new df
        encoded_eventlog_df = pandas.DataFrame(eventlog_df.iloc[:, 0])

        for column_name in eventlog_df:
            column_index = eventlog_df.columns.get_loc(column_name)

            # skip case ID
            if column_index > 0:

                column = eventlog_df[column_name]
                column_data_type = self.get_attribute_data_type(column)  # cat or num
                mode = self.get_encoding_mode(args, column_data_type)

                if column_index == 1:
                    # event ID
                    if self.data_structure['encoding']['is_doc2vec_set']:
                        encoded_column = self.encode_column(args, 'int', 'event', column_name)
                    else:
                        encoded_column = self.encode_column(args, mode, 'event', column_name)

                elif column_index == 2:
                    # timestamp
                    self.collect_time_deltas(column_name)
                    self.encode_time_deltas()
                    encoded_eventlog_df[column_name] = [time_delta for sublist in self.data_structure['encoding']['time_deltas'] for time_delta in sublist]
                    continue

                else:
                    # context attribute or outcome value
                    if column_index == (len(eventlog_df.columns) - 1):
                        attribute_type = 'outcome'
                    else:
                        attribute_type = 'context'

                    if self.data_structure['encoding']['is_doc2vec_set']:

                        if column_data_type == 'cat':
                            encoded_column = self.encode_column(args, 'int', attribute_type, column_name)
                        elif column_data_type == 'num':
                            encoded_column = self.encode_column(args, 'min_max_norm', attribute_type, column_name)

                    else:
                        encoded_column = self.encode_column(args, mode, attribute_type, column_name)

                encoded_eventlog_df = encoded_eventlog_df.join(encoded_column)

        encoded_eventlog_df.to_csv(self.data_structure['support']['encoded_data_dir'], sep=';', index=False)

        return encoded_eventlog_df

    def get_attribute_data_type(self, attribute_column):

        column_type = str(attribute_column.dtype)

        if column_type.startswith('float'):
            attribute_type = 'num'
        else:
            attribute_type = 'cat'

        return attribute_type

    def get_encoding_mode(self, args, data_type):

        if data_type == 'num':
            mode = args.encoding_num

        elif data_type == 'cat':
            mode = args.encoding_cat

        return mode

    def collect_time_deltas(self, column_name):

        process_instance_ids = self.data_structure['encoding']['eventlog_df'].iloc[:, 0]
        timestamps = self.data_structure['encoding']['eventlog_df'][column_name]
        tuples = list(zip(process_instance_ids, timestamps))
        time_deltas = []

        current_process_instance_id = process_instance_ids[0]
        predecessor_time = self.get_datetime_object(timestamps[0])
        time_deltas_process_instance = [0.0]

        for process_instance_id, timestamp in tuples[1:]:

            current_time = self.get_datetime_object(timestamp)

            if process_instance_id == current_process_instance_id:
                time_delta = self.calculate_time_difference(predecessor_time, current_time)

                if time_delta < 0:
                    time_delta = 0.0

                time_deltas_process_instance.append(time_delta)
            else:
                time_deltas.append(time_deltas_process_instance)
                time_deltas_process_instance = [0.0]
                current_process_instance_id = process_instance_id

            predecessor_time = current_time

        time_deltas.append(time_deltas_process_instance)
        self.data_structure['encoding']['time_deltas'] = time_deltas

        return

    def get_datetime_object(self, timestamp):

        datetime_object = datetime.datetime.strptime(timestamp, self.data_structure['support']['date_format'])

        return datetime_object

    def calculate_time_difference(self, predecessor_time, current_time):

        time_difference = (current_time - predecessor_time).total_seconds()

        return time_difference

    def encode_time_deltas(self):

        time_deltas_flat = [item for sublist in self.data_structure['encoding']['time_deltas'] for item in sublist]
        max_value = max(time_deltas_flat)
        encoded_time_deltas = [time_delta / max_value for sublist in self.data_structure['encoding']['time_deltas'] for
                               time_delta in sublist]

        element_idx = 0
        for sublist_idx, sublist in enumerate(self.data_structure['encoding']['time_deltas']):
            for sublist_element_idx in range(0, len(sublist)):
                self.data_structure['encoding']['time_deltas'][sublist_idx][sublist_element_idx] = encoded_time_deltas[
                    element_idx]
                element_idx += 1

    def encode_column(self, args, mode, attribute_type, attribute_name):

        if mode == 'int':
            encoded_column = self.apply_integer_mapping(attribute_type, attribute_name)

        elif mode == 'min_max_norm':
            encoded_column = self.apply_min_max_normalization(attribute_type, attribute_name)

        elif mode == 'onehot':
            encoded_column = self.apply_one_hot_encoding(attribute_type, attribute_name)

        elif mode == 'bin':
            encoded_column = self.apply_binary_encoding(attribute_type, attribute_name)

        elif mode == 'hash':
            encoded_column = self.apply_hash_encoding(args, attribute_type, attribute_name)

        else:
            # no encoding
            encoded_column = self.data_structure['encoding']['eventlog_df'][attribute_name]

        return encoded_column

    def apply_integer_mapping(self, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        data = dataframe[column_name].fillna("missing")
        unique_values = data.unique().tolist()
        int_mapping = dict(zip(unique_values, range(len(unique_values))))
        encoded_data = data.map(int_mapping)

        if attribute_type == 'event':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_event_encoding(len(encoded_data.columns))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_event_encoding(1)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        elif attribute_type == 'outcome':
            self.save_mapping_of_encoded_outcome_values(data, encoded_data)
            self.set_length_of_outcome_encoding(1)

        return encoded_data

    def apply_min_max_normalization(self, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        data = dataframe[column_name].fillna(dataframe[column_name].mean())

        encoded_data = (data - data.min()) / (data.max() - data.min())

        if attribute_type == 'context':
            self.set_length_of_context_encoding(1)
        elif attribute_type == 'outcome':
            self.set_length_of_outcome_encoding(1)

        return encoded_data

    def apply_one_hot_encoding(self, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        onehot_encoder = category_encoders.OneHotEncoder(cols=[column_name])
        encoded_df = onehot_encoder.fit_transform(dataframe)

        encoded_data = encoded_df[
            encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        if attribute_type == 'event':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_event_encoding(len(encoded_data.columns))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_event_encoding(1)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        elif attribute_type == 'outcome':
            self.save_mapping_of_encoded_outcome_values(dataframe[column_name], encoded_data)
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_outcome_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_outcome_encoding(1)

        return encoded_data

    def apply_binary_encoding(self, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        binary_encoder = category_encoders.BinaryEncoder(cols=[column_name])
        encoded_df = binary_encoder.fit_transform(dataframe)

        encoded_data = encoded_df[
            encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        if attribute_type == 'event':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_event_encoding(len(encoded_data.columns))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_event_encoding(1)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        elif attribute_type == 'outcome':
            self.save_mapping_of_encoded_outcome_values(dataframe[column_name], encoded_data)
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_outcome_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_outcome_encoding(1)

        return encoded_data

    def apply_hash_encoding(self, args, attribute_type, column_name):

        dataframe = self.data_structure['encoding']['eventlog_df']

        dataframe[column_name] = dataframe[column_name].fillna("missing")
        hash_encoder = category_encoders.HashingEncoder(cols=[column_name], n_components=args.num_hash_output)
        encoded_df = hash_encoder.fit_transform(dataframe)

        encoded_data = encoded_df[encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith('col_')]]

        new_column_names = []
        for number in range(len(encoded_df.columns)):
            new_column_names.append(column_name + "_%d" % number)

        encoded_data = encoded_data.rename(columns=dict(zip(encoded_df.columns.tolist(), new_column_names)))

        if attribute_type == 'event':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_event_encoding(len(encoded_data.columns))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_event_encoding(1)

        elif attribute_type == 'context':
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_context_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_context_encoding(1)

        elif attribute_type == 'outcome':
            self.save_mapping_of_encoded_outcome_values(dataframe[column_name], encoded_data)
            if isinstance(encoded_data, pandas.DataFrame):
                self.set_length_of_outcome_encoding(len(encoded_data.columns.tolist()))
            elif isinstance(encoded_data, pandas.Series):
                self.set_length_of_outcome_encoding(1)

        return encoded_data

    def save_mapping_of_encoded_outcome_values(self, column, encoded_column):

        encoded_column_tuples = []
        for entry in encoded_column.values.tolist():

            if type(entry) != list:
                encoded_column_tuples.append((entry,))
            else:
                encoded_column_tuples.append(tuple(entry))

        tuple_all_rows = list(zip(column.values.tolist(), encoded_column_tuples))

        tuple_unique_rows = []
        for tuple_row in tuple_all_rows:

            if tuple_row not in tuple_unique_rows:
                tuple_unique_rows.append(tuple_row)

        mapping = dict(tuple_unique_rows)

        self.data_structure['encoding']['mapping_outcome_values'] = mapping

    def set_length_of_event_encoding(self, num_columns):
        self.data_structure['encoding']['num_values_event_ids'] = num_columns

    def set_length_of_context_encoding(self, num_columns):
        self.data_structure['encoding']['num_values_context_list'].append(num_columns)

    def set_length_of_outcome_encoding(self, num_columns):
        self.data_structure['encoding']['num_values_outcome'] = num_columns

    def get_sequences_from_encoded_eventlog(self, eventlog_df):

        utils.llprint("Create instances ... \n")

        id_latest_process_instance = ''
        first_event_of_process_instance = True
        process_instance = []
        context_attributes_process_instance = []
        outcome_values_process_instance = []
        output = True

        for index, event in eventlog_df.iterrows():

            id_current_process_instance = int(event[0])

            if output:
                self.check_for_context_attributes_df(event)
                output = False

            if id_current_process_instance != id_latest_process_instance:
                self.add_data_to_data_structure(id_current_process_instance, 'ids_process_instances')

                if not first_event_of_process_instance:

                    self.add_data_to_data_structure(process_instance, 'process_instances')
                    self.add_data_to_data_structure(outcome_values_process_instance, 'outcome_values')
                    if self.data_structure['meta']['num_attributes_context'] > 0:
                        self.add_data_to_data_structure(context_attributes_process_instance,
                                                        'context_attributes_process_instances')

                process_instance = []
                outcome_values_process_instance = []

                if self.data_structure['meta']['num_attributes_context'] > 0:
                    context_attributes_process_instance = []

                self.data_structure['meta']['num_process_instances'] += 1

            process_instance = self.add_encoded_event_to_process_instance(event, process_instance)
            outcome_value_of_event = self.get_outcome_value_of_event(event)
            outcome_values_process_instance.append(outcome_value_of_event)
            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_event = self.get_context_attributes_of_event(event)
                context_attributes_process_instance.append(context_attributes_event)

            id_latest_process_instance = id_current_process_instance
            first_event_of_process_instance = False

        self.add_data_to_data_structure(process_instance, 'process_instances')
        self.add_data_to_data_structure(outcome_values_process_instance, 'outcome_values')

        if self.data_structure['meta']['num_attributes_context'] > 0:
            self.add_data_to_data_structure(context_attributes_process_instance, 'context_attributes_process_instances')

        self.data_structure['meta']['num_process_instances'] += 1

    def check_for_context_attributes_df(self, event):

        if len(event) == self.data_structure['encoding']['num_values_control_flow'] + self.data_structure['encoding']['num_values_outcome']:
            utils.llprint("No context attributes found ...\n")
        else:
            self.data_structure['meta']['num_attributes_context'] = len(
                self.data_structure['encoding']['num_values_context_list'])
            self.data_structure['encoding']['num_values_context'] = sum(
                self.data_structure['encoding']['num_values_context_list'])
            utils.llprint("%d context attributes found ...\n" % self.data_structure['meta']['num_attributes_context'])

    def add_encoded_event_to_process_instance(self, event, process_instance):

        encoded_event_id = []
        start_index = 1
        end_index = self.data_structure['encoding']['num_values_event_ids'] + 1

        for enc_val in range(start_index, end_index):
            encoded_event_id.append(event[enc_val])

        process_instance.append(tuple(encoded_event_id))

        return process_instance

    def get_context_attributes_of_event(self, event):

        event = event.tolist()
        context_attributes_event = []

        for context_attribute_index in range(self.data_structure['encoding']['num_values_control_flow'],
                                             self.data_structure['encoding']['num_values_control_flow'] +
                                             self.data_structure['encoding']['num_values_context']):
            context_attributes_event.append(event[context_attribute_index])

        return context_attributes_event

    def get_outcome_value_of_event(self, event):

        event = event.tolist()
        outcome_value_event = []

        for outcome_value_index in range(self.data_structure['encoding']['num_values_control_flow'] + self.data_structure['encoding']['num_values_context'], len(event)):
            outcome_value_event.append(event[outcome_value_index])

        return tuple(outcome_value_event)

    def add_data_to_data_structure(self, values, structure):

        self.data_structure['data'][structure].append(values)

    def get_training_set(self):

        utils.llprint("Get training instances ... \n")
        process_instances_train, context_attributes_train, outcome_values_train, _, time_deltas_train = self.set_instances_of_fold('train')

        utils.llprint("Create cropped training instances ... \n")

        cropped_process_instances, cropped_context_attributes, cropped_outcome_values, cropped_time_deltas, next_events = \
            self.get_cropped_instances(
                process_instances_train,
                context_attributes_train,
                outcome_values_train,
                time_deltas_train)

        return cropped_process_instances, cropped_time_deltas, cropped_context_attributes, cropped_outcome_values, next_events

    def set_indices_k_fold_validation(self):
        """ Produces indices for each fold of a k-fold cross-validation. """

        k_fold = KFold(n_splits=self.data_structure['support']['num_folds'], random_state=0, shuffle=False)

        for train_indices, test_indices in k_fold.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    def set_indices_split_validation(self, args):
        """ Produces indices for train and test set of a split-validation. """

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test, random_state=0)

        for train_indices, test_indices in shuffle_split.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    def set_instances_of_fold(self, mode):
        """ Retrieves instances of a fold. """

        process_instances_of_fold = []
        ids_process_instances_of_fold = []
        context_attributes_of_fold = []
        time_deltas_of_fold = []
        outcome_values_of_fold = []

        for value in self.data_structure['support'][mode + '_index_per_fold'][
            self.data_structure['support']['iteration_cross_validation']]:
            process_instances_of_fold.append(self.data_structure['data']['process_instances'][value])
            ids_process_instances_of_fold.append(self.data_structure['data']['ids_process_instances'][value])
            time_deltas_of_fold.append(self.data_structure['encoding']['time_deltas'][value])
            outcome_values_of_fold.append(self.data_structure['data']['outcome_values'][value])

            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_of_fold.append(
                    self.data_structure['data']['context_attributes_process_instances'][value])

        if mode == 'train':
            self.data_structure['data']['train']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['train']['context_attributes'] = context_attributes_of_fold
            self.data_structure['data']['train']['outcome_values'] = outcome_values_of_fold
            self.data_structure['data']['train']['ids_process_instances'] = ids_process_instances_of_fold
            self.data_structure['data']['train']['time_deltas'] = time_deltas_of_fold

        elif mode == 'test':
            self.data_structure['data']['test']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['test']['context_attributes'] = context_attributes_of_fold
            self.data_structure['data']['test']['outcome_values'] = outcome_values_of_fold
            self.data_structure['data']['test']['ids_process_instances'] = ids_process_instances_of_fold
            self.data_structure['data']['test']['time_deltas'] = time_deltas_of_fold

        return process_instances_of_fold, context_attributes_of_fold, outcome_values_of_fold, ids_process_instances_of_fold, time_deltas_of_fold

    def get_cropped_instances(self, process_instances, context_attributes_process_instances,
                              outcome_values, time_deltas_process_instances):
        """ Crops prefixes out of instances. """

        cropped_process_instances = []
        cropped_time_deltas = []
        cropped_context_attributes = []
        cropped_outcome_values = []
        next_events = []

        if self.data_structure['meta']['num_attributes_context'] > 0:

            for process_instance, time_deltas_process_instance, context_attributes_process_instance, outcome_value in zip(
                    process_instances, time_deltas_process_instances,
                    context_attributes_process_instances, outcome_values):

                for i in range(0, len(process_instance)):

                    if i == 0:
                        continue

                    # 0:i -> get 0 up to n-1 events of a process instance, since n is the label
                    cropped_process_instances.append(process_instance[0:i])
                    cropped_time_deltas.append(time_deltas_process_instance[0:i])
                    cropped_context_attributes.append(context_attributes_process_instance[0:i])
                    cropped_outcome_values.append(outcome_value[0:i])
                    # label
                    next_events.append(process_instance[i])
        else:
            for process_instance, time_delta_process_instance, outcome_value in zip(
                    process_instances, time_deltas_process_instances, outcome_values):

                for i in range(0, len(process_instance)):

                    if i == 0:
                        continue
                    cropped_process_instances.append(process_instance[0:i])
                    cropped_time_deltas.append(time_delta_process_instance[0:i])
                    cropped_outcome_values.append(outcome_value[0:i])
                    # label
                    next_events.append(process_instance[i])

        return cropped_process_instances, cropped_context_attributes, cropped_outcome_values, cropped_time_deltas, next_events

    def get_cropped_instance(self, prefix_size, index, process_instance, time_deltas, outcome_value):
        """ Crops prefixes out of a single process instance. """

        cropped_process_instance = process_instance[:prefix_size]
        cropped_time_deltas = time_deltas[:prefix_size]
        cropped_outcome_values = outcome_value[:prefix_size]
        if self.data_structure['meta']['num_attributes_context'] > 0:
            cropped_context_attributes = self.data_structure['data']['test']['context_attributes'][index][:prefix_size]
        else:
            cropped_context_attributes = []

        return cropped_process_instance, cropped_time_deltas, cropped_context_attributes, cropped_outcome_values

    def get_data_tensor(self, args, cropped_process_instances, cropped_time_deltas,
                        cropped_context_attributes_process_instance, cropped_outcome_values,
                        mode, data_structure):
        """ Produces a vector-oriented representation of data as 3-dimensional tensor.
            Note batches of the training set are created by a data generator,
            so we consider the data structure as a closed object.
        """

        if data_structure['encoding']['is_doc2vec_set']:
            data_set = self.get_data_tensor_from_doc2vec(args, cropped_process_instances, cropped_time_deltas,
                                                         cropped_context_attributes_process_instance,
                                                         cropped_outcome_values, mode, data_structure)

            return data_set

        else:

            if args.include_time_delta:

                if mode == 'train':

                    data_set = numpy.zeros((
                        len(cropped_process_instances),
                        data_structure['meta']['max_length_process_instance'],
                        data_structure['encoding']['num_values_event_ids'] + 1 + data_structure['encoding'][
                            'num_values_context'] + data_structure['encoding']['num_values_outcome']), dtype=numpy.float64)
                else:
                    data_set = numpy.zeros((
                        1,
                        data_structure['meta']['max_length_process_instance'],
                        data_structure['encoding']['num_values_event_ids'] + 1 + data_structure['encoding'][
                            'num_values_context'] + data_structure['encoding']['num_values_outcome']), dtype=numpy.float32)

                for index, (cropped_process_instance, cropped_time_deltas_process_instance, cropped_outcome_value) in enumerate(
                        zip(cropped_process_instances, cropped_time_deltas, cropped_outcome_values)):

                    left_pad = data_structure['meta']['max_length_process_instance'] - len(cropped_process_instance)

                    if data_structure['meta']['num_attributes_context'] > 0:
                        cropped_context_attributes = cropped_context_attributes_process_instance[index]

                    for time_step, (event, time_delta, outcome_value) in enumerate(
                            zip(cropped_process_instance, cropped_time_deltas_process_instance, cropped_outcome_value)):

                        for tuple_idx in range(0, data_structure['encoding']['num_values_event_ids']):
                            data_set[index, time_step + left_pad, tuple_idx] = event[tuple_idx]

                        data_set[index, time_step + left_pad, data_structure['encoding']['num_values_event_ids']] = time_delta

                        if data_structure['meta']['num_attributes_context'] > 0:
                            for context_attribute_index in range(0, data_structure['encoding']['num_values_context']):
                                data_set[index, time_step + left_pad, data_structure['encoding']['num_values_event_ids'] + 1 + context_attribute_index] = cropped_context_attributes[time_step][
                                    context_attribute_index]

                        for tuple_index in range(0, data_structure['encoding']['num_values_outcome']):
                            data_set[index, time_step + left_pad, data_structure['encoding']['num_values_event_ids'] + 1 + data_structure['encoding']['num_values_context'] +
                                     tuple_index] = outcome_value[tuple_index]

                return data_set

            else:

                if mode == 'train':
                        data_set = numpy.zeros((
                            len(cropped_process_instances),
                            data_structure['meta']['max_length_process_instance'],
                            data_structure['encoding']['num_values_event_ids'] + data_structure['encoding'][
                                'num_values_context'] + data_structure['encoding']['num_values_outcome']), dtype=numpy.float64)
                else:
                    data_set = numpy.zeros((
                        1,
                        data_structure['meta']['max_length_process_instance'],
                        data_structure['encoding']['num_values_event_ids']+ data_structure['encoding'][
                            'num_values_context'] + data_structure['encoding']['num_values_outcome']), dtype=numpy.float32)

                for index, (cropped_process_instance, cropped_time_deltas_process_instance, cropped_outcome_value) in enumerate(
                        zip(cropped_process_instances, cropped_time_deltas, cropped_outcome_values)):

                    left_pad = data_structure['meta']['max_length_process_instance'] - len(cropped_process_instance)

                    if data_structure['meta']['num_attributes_context'] > 0:
                        cropped_context_attributes = cropped_context_attributes_process_instance[index]

                    for time_step, (event, outcome_value) in enumerate(
                            zip(cropped_process_instance, cropped_outcome_value)):

                        for tuple_idx in range(0, data_structure['encoding']['num_values_event_ids']):
                            data_set[index, time_step + left_pad, tuple_idx] = event[tuple_idx]

                        if data_structure['meta']['num_attributes_context'] > 0:
                            for context_attribute_index in range(0, data_structure['encoding']['num_values_context']):
                                data_set[index, time_step + left_pad, data_structure['encoding']['num_values_event_ids'] + context_attribute_index] = cropped_context_attributes[time_step][
                                    context_attribute_index]

                        for tuple_index in range(0, data_structure['encoding']['num_values_outcome']):
                            data_set[index, time_step + left_pad, data_structure['encoding']['num_values_event_ids'] + data_structure['encoding']['num_values_context'] +
                                     tuple_index] = outcome_value[tuple_index]

                return data_set

    def get_data_tensor_from_doc2vec(self, args, cropped_process_instances, cropped_time_deltas,
                                     cropped_context_attributes, cropped_outcome_values, mode, data_structure):

        event_sentences = self.collect_event_sentences_for_doc2vec(cropped_process_instances, cropped_time_deltas,
                                                                   cropped_context_attributes, cropped_outcome_values, data_structure)

        encoded_event_sentences = self.apply_doc2vec_encoding(args, event_sentences, data_structure)

        if mode == 'train':
            data_set = numpy.zeros((
                len(cropped_process_instances),
                data_structure['meta']['max_length_process_instance'],
                args.doc2vec_vec_size), dtype=numpy.float64)

        else:
            data_set = numpy.zeros((
                1,
                data_structure['meta']['max_length_process_instance'],
                args.doc2vec_vec_size), dtype=numpy.float32)

        encoded_event_index = 0
        for index, cropped_process_instance in enumerate(cropped_process_instances):

            left_pad = data_structure['meta']['max_length_process_instance'] - len(cropped_process_instance)

            for time_step in range(len(cropped_process_instance)):
                data_set[index, time_step + left_pad] = encoded_event_sentences[encoded_event_index]
                encoded_event_index += 1

        return data_set

    def build_doc2vec_model(self, args):

        utils.llprint("Build and train Doc2Vec model ... \n")

        # collect data
        document = open(self.data_structure['support']['encoded_data_dir'], 'r')
        doc_reader = csv.reader(document, delimiter=';', quotechar='|')
        next(doc_reader)

        data = []
        event_sentences = []
        for row in doc_reader:
            data.append(row)
            event_sentences.append(row[1:])

        # tag data
        tagged_data = [gensim.models.doc2vec.TaggedDocument(
            words=sentence, tags=[str(index)]) for index, sentence in enumerate(event_sentences)]

        # initialize model
        model = gensim.models.Doc2Vec(tagged_data,
                                          dm=0,
                                          vector_size=args.doc2vec_vec_size,
                                          window=5,
                                          min_count=1,
                                          alpha=args.doc2vec_alpha,
                                          min_alpha=0.025,
                                          worker=12,
                                          epochs=args.doc2vec_num_epochs)

        self.train_and_save_doc2vec_model(args, model, tagged_data)

    def train_and_save_doc2vec_model(self, args, model, tagged_data):

        for epoch in range(args.doc2vec_num_epochs):
            if epoch % 2 == 0:
                 utils.llprint("Doc2vec training in epoch %s ...\n" % epoch)
            model.train(tagged_data, total_examples=len(tagged_data), epochs=args.doc2vec_num_epochs)
            model.alpha -= 0.002
            model.min_alpha = model.alpha

            model.save(args.checkpoint_dir + str(0) + '_events_doc2vec_2d' + str(args.doc2vec_vec_size) + '.model',
                       sep_limit=2000000000)
            self.data_structure['support']['doc2vec_model'] = model

    def collect_event_sentences_for_doc2vec(self, cropped_process_instances, cropped_time_deltas,
                                            cropped_context_attributes, cropped_outcome_values, data_structure):

        event_sentence = []
        event_sentences_process_instance = []
        event_sentences_process_instances = []

        if data_structure['meta']['num_attributes_context'] > 0:
            for cropped_process_instance, cropped_time_deltas_process_instance, \
                cropped_context_attributes_process_instance, cropped_outcome_values_process_instance in zip(
                    cropped_process_instances, cropped_time_deltas, cropped_context_attributes, cropped_outcome_values):
                for event, time_delta, context_attributes, outcome_value in zip(cropped_process_instance,
                                                                 cropped_time_deltas_process_instance,
                                                                 cropped_context_attributes_process_instance,
                                                                 cropped_outcome_values_process_instance):
                    event_sentence.append(event)
                    event_sentence.append(time_delta)
                    event_sentence.append(context_attributes)
                    event_sentence.append(outcome_value)

                    event_sentences_process_instance.append(event_sentence)
                    event_sentence = []

                event_sentences_process_instances.append(event_sentences_process_instance)
                event_sentences_process_instance = []
        else:
            for cropped_process_instance, cropped_time_deltas_process_instance, \
                cropped_outcome_values_process_instance in zip(cropped_process_instances, cropped_time_deltas,
                                                               cropped_outcome_values):

                for event, time_delta, outcome_value in zip(cropped_process_instance, cropped_time_deltas_process_instance,
                                                            cropped_outcome_values_process_instance):
                    event_sentence.append(event)
                    event_sentence.append(time_delta)
                    event_sentence.append(outcome_value)

                    event_sentences_process_instance.append(event_sentence)
                    event_sentence = []

                event_sentences_process_instances.append(event_sentences_process_instance)
                event_sentences_process_instance = []

        return event_sentences_process_instances

    def apply_doc2vec_encoding(self, args, data, data_structure):

        model = self.data_structure['support']['doc2vec_model']
        # model = self.load_doc2vec_model(args)
        prepared_data = self.prepare_data_for_doc2vec(args, data, data_structure)
        tagged_data = self.tag_data_for_doc2vec(prepared_data)

        encoded_data = []
        for event_sentence in tagged_data:
            vector = model.infer_vector(event_sentence.words)
            encoded_data.append(vector)

        return encoded_data

    def load_doc2vec_model(self, args):

        model = gensim.models.Doc2Vec.load(
            args.checkpoint_dir + str(0) + '_events_doc2vec_2d' + str(args.doc2vec_vec_size) + '.model')

        return model

    def prepare_data_for_doc2vec(self, args, data, data_structure):

        event_sentences = []

        for process_instance_list in data:

            for event_list in process_instance_list:

                # note each attribute represents one column
                event = []
                event_id_tuple = event_list[0]
                event_id = event_id_tuple[0]
                time_delta = event_list[1]

                if data_structure['meta']['num_attributes_context'] > 0:
                    context_attributes_values = event_list[2]

                event.append(str(event_id))

                if args.include_time_delta:
                    event.append(str(time_delta))

                if data_structure['meta']['num_attributes_context'] > 0:
                    for value in context_attributes_values:
                        event.append(str(value))

                outcome_values = event_list[-1]
                event.append(str(outcome_values))

                event_sentences.append(event)

        return event_sentences

    def tag_data_for_doc2vec(self, event_sentences):

        tagged_data = [gensim.models.doc2vec.TaggedDocument(words=sentence, tags=[str(index)]) for index, sentence in
                       enumerate(event_sentences)]

        return tagged_data

    def get_data_tensor_for_single_prediction(self, args, cropped_process_instance, cropped_time_deltas,
                                              cropped_context_attributes, cropped_outcome_value, data_structure):

        data_set = self.get_data_tensor(
            args,
            [cropped_process_instance],
            [cropped_time_deltas],
            [cropped_context_attributes],
            [cropped_outcome_value],
            'test',
            data_structure
        )

        return data_set

    def get_label_tensor(self, cropped_process_instances, outcome_values, data_structure):
        """ Produces a vector-oriented representation of label as 2-dimensional tensor. """

        label = numpy.zeros((len(cropped_process_instances), len(data_structure['support']['outcome_types'])),
                            dtype=numpy.float64)

        for index, cropped_process_instance in enumerate(cropped_process_instances):

            for outcome_type in data_structure['support']['outcome_types']:

                if outcome_type == outcome_values[index]:
                    label[index, data_structure['support']['map_outcome_type_to_outcome_value'][outcome_type]] = 1
                else:
                    label[index, data_structure['support']['map_outcome_type_to_outcome_value'][outcome_type]] = 0

        return label

    def get_outcome_type(self, predictions):

        max_prediction = 0
        outcome_type = ''
        index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                outcome_type = self.data_structure['support']['map_outcome_value_to_outcome_type'][index]
            index += 1

        return outcome_type
