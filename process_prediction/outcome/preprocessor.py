from __future__ import division
import numpy
import process_prediction.utils as utils
from sklearn.model_selection import KFold, ShuffleSplit
import pandas
import process_prediction.utils as utils
import category_encoders
from pm4py.objects.conversion.log import converter as log_converter
import datetime


class Preprocessor(object):

    iteration_cross_validation = 0
    activity = {}
    context = {}
    classes = {}

    def __init__(self):

        utils.llprint("Initialization ... \n")
        self.activity = {
            'label_length': 0
        }
        self.context = {
            'attributes': [],
            'data_types': [],
            'encoding_lengths': []
        }
        self.classes = {
            'labels': {},
            'ids_to_labels': {},
            'labels_to_ids': {},
        }


    def get_event_log(self, args):
        """
        Constructs an event log from a csv file using PM4PY

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.

        Returns
        -------
        list of dicts, where single dict represents a case :
            pm4py.objects.log.log.EventLog object representing an event log.

        """

        df = self.load_data(args)

        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: args.case_id_key}
        event_log = log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        self.set_classes(args, event_log)

        df_enc = self.encode_data(args, df)
        event_log_enc = log_converter.apply(df_enc, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

        return event_log_enc

    def load_data(self, args):
        """
        Retrieves event log data from a csv-file and returns it as a dataframe.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.

        Returns
        -------
        pandas.DataFrame :
            Dataframe representing the event log.

        """

        utils.llprint('Load data ... \n')

        df = pandas.read_csv(args.data_dir + args.data_set, sep=',')
        self.set_context(df)

        return df

    def set_context(self, df):
        """
        Retrieves names and data types of context attributes in an event log.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe representing the event log.

        Returns
        -------
        None

        """

        attributes = []
        data_types = []
        column_names = df.columns
        for idx, attribute_name in enumerate(column_names):
            if idx > 2 and idx < len(column_names)-1:
                attributes.append(attribute_name)
                data_types.append(self.get_attribute_data_type(df[attribute_name]))
        self.context['attributes'] = attributes
        self.context['data_types'] = data_types

    def get_context_attributes(self):
        """
        Returns the names of the context attributes in the event log.

        Returns
        -------
        list of str :
            Names of the context attributes in the event log.

        """
        return self.context['attributes']

    def encode_data(self, args, df):
        """
        Encodes an event log represented by a dataframe.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        df : pandas.DataFrame
            Dataframe representing the event log.

        Returns
        -------
        pandas.DataFrame :
            Encoded dataframe representing the event log.

        """

        utils.llprint('Encode data ... \n')

        encoded_df = pandas.DataFrame(df.iloc[:, 0])

        for column_name in df:
            column_index = df.columns.get_loc(column_name)

            if column_index == 0:
                # no encoding of case id
                encoded_df[column_name] = df[column_name]
            else:
                if column_index == 1:
                    encoded_column = self.encode_activities(args, df, column_name)
                elif column_index == 2 or column_index == len(df.columns)-1:
                    # timestamps and outcome, no encoding # TODO outcome no encoding?
                    encoded_df[column_name] = df[column_name]
                    if column_index == 2:
                        # time deltas (time difference between event and predecessor of the same case)
                        time_deltas = self.get_time_deltas(args, df, column_name)
                        encoded_df[args.time_delta_key] = [time_delta for sublist in time_deltas for time_delta in sublist]
                    continue
                elif column_index > 2:
                    encoded_column = self.encode_context_attribute(args, df, column_name)
                encoded_df = encoded_df.join(encoded_column)

        self.export_encoded_data(args, encoded_df)

        return encoded_df

    def export_encoded_data(self, args, encoded_df):
        """
        Exports the encoded dataframe as a csv file.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        encoded_df : pandas.DataFrame
            Encoded dataframe representing the event log.

        Returns
        -------
        None

        """

        encoded_df.to_csv(r'%s' % args.data_dir + r'encoded_%s' % args.data_set, sep=';', index=False)

    def get_attribute_data_type(self, attribute_column):
        """
        Returns the data type of the passed attribute column.

        Parameters
        ----------
        attribute_column : pandas.Series
            An attribute column of the dataframe.

        Returns
        -------
        str :
            Indicates the data type of the values of the attribute column.

        """

        column_type = str(attribute_column.dtype)

        if column_type.startswith('float'):
            attribute_type = 'num'
        else:
            attribute_type = 'cat'

        return attribute_type

    def get_encoding_mode(self, args, data_type):
        """
        Returns the encoding method to be used for a given data type as specified in config file.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        data_type : str
            Indicates the data type of the values of the attribute column.

        Returns
        -------
        str :
            Encoding method.

        """

        if data_type == 'num':
            mode = args.encoding_num
        elif data_type == 'cat':
            mode = args.encoding_cat

        return mode

    def get_time_deltas(self, args, df, column_name):
        """
        Calculates time differences between two successive events of the same case.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        df : pandas.DataFrame
            Dataframe representing the event log.
        column_name : str
            Name of the dataframe column.

        Returns
        -------
        list (of lists) :
            A sublist contains calculated time differences between two successive events of the same case.

        """

        case_ids = df[args.case_id_key]
        timestamps = df[column_name]
        tuples = list(zip(case_ids, timestamps))
        time_deltas = []

        current_case_id = case_ids[0]
        predecessor_time = self.get_datetime_object(args, timestamps[0])
        time_deltas_case = [0.0]

        for case_id, timestamp in tuples[1:]:

            current_time = self.get_datetime_object(args, timestamp)

            if case_id == current_case_id:
                time_delta = self.get_time_difference(predecessor_time, current_time)

                if time_delta < 0:
                    time_delta = 0.0

                time_deltas_case.append(time_delta)
            else:
                time_deltas.append(time_deltas_case)
                time_deltas_case = [0.0]
                current_case_id = case_id

            predecessor_time = current_time

        time_deltas.append(time_deltas_case)

        norm_time_deltas = self.normalize_time_deltas(time_deltas)

        return norm_time_deltas

    def get_datetime_object(self, args, timestamp):
        """
        Creates datetime objects from the timestamp of an event.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        timestamp : str
            The timestamp of an event.

        Returns
        -------
        datetime :
            Datetime object from the timestamp of an event.

        """

        datetime_object = datetime.datetime.strptime(timestamp, args.date_format)

        return datetime_object

    def get_time_difference(self, predecessor_time, current_time):
        """
        Calculates the time difference between two (successive) events.

        Parameters
        ----------
        predecessor_time : datetime
            Datetime object of an event.
        current_time : datetime
            Datetime object of the following event.

        Returns
        -------
        float :
            The time difference.

        """

        time_difference = (current_time - predecessor_time).total_seconds()

        return time_difference

    def normalize_time_deltas(self, time_deltas):
        """
        Normalize the computed time differences so that each value is between 0 and 1.

        Parameters
        ----------
        time_deltas : list (of lists)
            A sublist contains the calculated time differences  between two successive events of the same case.

        Returns
        -------
        list (of lists)
            A sublist contains the relative/normalized time differences between two successive events of the same case.
            The relative/normalized time difference is the absolute time difference divided by the highest time
            difference over all cases. The resulting relative/normalized time differences are always between 0 and 1.

        """

        time_deltas_flat = [item for sublist in time_deltas for item in sublist]
        max_value = max(time_deltas_flat)
        encoded_time_deltas = [time_delta / max_value for sublist in time_deltas for time_delta in sublist]

        element_idx = 0
        for sublist_idx, sublist in enumerate(time_deltas):
            for sublist_element_idx in range(0, len(sublist)):
                time_deltas[sublist_idx][sublist_element_idx] = encoded_time_deltas[element_idx]
                element_idx += 1

        return time_deltas

    def encode_activities(self, args, df, column_name):
        """
        Encodes activity for each event in an event log.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        df : pandas.DataFrame
            Dataframe representing the event log.
        column_name : str
            Name of the dataframe column / attribute to be encoded.

        Returns
        -------
        pandas.Series :
            Each entry is a list which represents an encoded activity (= label).

        """

        encoding_mode = args.encoding_cat
        encoding_columns = self.encode_column(df, column_name, encoding_mode)

        if isinstance(encoding_columns, pandas.DataFrame):
            self.set_length_of_activity_encoding(len(encoding_columns.columns))
        elif isinstance(encoding_columns, pandas.Series):
            self.set_length_of_activity_encoding(1)

        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)

        return df[column_name]

    def encode_context_attribute(self, args, df, column_name):
        """
        Encodes values of a context attribute for all events in an event log.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        df : pandas.DataFrame
            Dataframe representing the event log.
        column_name : str
            Name of the dataframe column / attribute to be encoded.

        Returns
        -------
        pandas.Series :
            Each entry represents an encoded context attribute value.
            Entries can be lists or a float (if encoding mode is min-max-normalization)

        """

        data_type = self.get_attribute_data_type(df[column_name])
        encoding_mode = self.get_encoding_mode(args, data_type)

        encoding_columns = self.encode_column(df, column_name, encoding_mode)
        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)

        if isinstance(encoding_columns, pandas.DataFrame):
            self.set_length_of_context_encoding(len(encoding_columns.columns))
        elif isinstance(encoding_columns, pandas.Series):
            self.set_length_of_context_encoding(1)

        return df[column_name]

    def encode_column(self, df, column_name, mode):
        """
        Returns columns containing encoded values for a given attribute column.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe representing the event log.
        column_name : str
            Name of the dataframe column.
        mode : str
            Encoding method.

        Returns
        -------
        pandas.DataFrame :
            Encoding of the column.

        """

        if mode == 'min_max_norm':
            encoding_columns = self.apply_min_max_normalization(df, column_name)

        elif mode == 'onehot':
            encoding_columns = self.apply_one_hot_encoding(df, column_name)

        else:
            # no encoding
            encoding_columns = df[column_name]

        return encoding_columns

    def apply_min_max_normalization(self, df, column_name):
        """
        Normalizes a dataframe column with min-max normalization.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe representing the event log.
        column_name : str
            Name of the dataframe column.

        Returns
        -------
        pandas.DataFrame :
            Encoding of the column.

        """

        column = df[column_name].fillna(df[column_name].mean())
        encoded_column = (column - column.min()) / (column.max() - column.min())

        return encoded_column

    def apply_one_hot_encoding(self, df, column_name):
        """
        Encodes a dataframe column with one hot encoding.

        df : pandas.DataFrame
            Dataframe representing the event log.
        column_name : str
            Name of the dataframe column.

        Returns
        -------
        pandas.DataFrame :
            Encoding of the column.

        """

        onehot_encoder = category_encoders.OneHotEncoder(cols=[column_name])
        encoded_df = onehot_encoder.fit_transform(df)

        encoded_column = encoded_df[encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        return encoded_column

    def set_classes(self, args, event_log):
        """
        Creates a mapping for classes (ids + labels).

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.

        event_log : list of dicts, where single dict represents a case
            The event log.

        Returns
        -------
        None

        """
        labels = list()
        map_class_labels_to_ids = dict()
        map_class_ids_to_labels = dict()

        for case in event_log:
            for event in case._list:
                labels.append(event[args.outcome_key])

        for index, value in enumerate(list(set(labels))):
            map_class_labels_to_ids[value] = index
            map_class_ids_to_labels[index] = value

        self.classes['labels'] = list(set(labels))
        self.classes['ids_to_labels'] = map_class_labels_to_ids
        self.classes['labels_to_ids'] = map_class_ids_to_labels

    def transform_encoded_attribute_columns_to_single_column(self, encoded_columns, df, column_name):
        """
        Transforms multiple columns (repr. encoded attribute) to a single column in a dataframe.

        Parameters
        ----------
        encoded_columns : pandas.Dataframe
            Encoding of the initial column.
        df : pandas.DataFrame
            Dataframe representing the event log.
        column_name : str
            Name of the initial dataframe column.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe representing the (partially) encoded event log.

        """

        encoded_values_list = encoded_columns.values.tolist()
        df[column_name] = encoded_values_list

        return df

    def set_length_of_activity_encoding(self, num_columns):
        """
        Saves number of values representing an encoded activity (= label).

        Parameters
        ----------
        num_columns : int
            The number of columns / values used to represent an encoded activity.

        Returns
        -------
        None

        """
        self.activity['label_length'] = num_columns

    def set_length_of_context_encoding(self, num_columns):
        """
        Saves number of values representing an encoded context attribute.

        Parameters
        ----------
        num_columns : int
            The number of columns / values used to represent an encoded context attribute.

        Returns
        -------
        None

        """
        self.context['encoding_lengths'].append(num_columns)

    def get_length_of_activity_label(self):
        """
        Returns number of values representing an encoded activity

        Returns
        -------
        int :
            The number of values used to represent an encoded activity (= label)

        """
        return self.activity['label_length']

    def get_lengths_of_context_encoding(self):
        """
        Returns number of values representing an encoded context attribute.

        Returns
        -------
        list of ints :
        The number of values used to represent encoded context attributes.
            First int is length of first context attribute, second int is length of second context attribute, ...

        """
        return self.context['encoding_lengths']

    def get_class_labels(self):
        """
        Returns labels representing encoded classes of an event log.

        Returns
        -------
        list of tuples :
            Tuples represent labels (= encoded classes).

        """
        return self.classes['labels']

    def get_num_classes(self):
        """
        Returns the number of prediction classes occurring in the event log.

        Returns
        -------
        int :
            Number of classes.

        """
        return len(self.get_class_labels())

    def context_exists(self):
        """
        Checks whether context attributes exist.

        Returns
        -------
        bool :
            Indicates if context attributes exist or not.

        """
        return len(self.get_context_attributes()) > 0

    def get_num_features(self, args):
        """
        Returns the number of features used to train and test the model.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.

        Returns
        -------
        int :
            The sum of the number of values of all encoded attributes' representations.

        """

        num_features = 0
        num_features += self.get_length_of_activity_label()
        for len in self.get_lengths_of_context_encoding():
            num_features += len

        if args.include_time_delta:
            num_features += 1

        return num_features

    def get_max_case_length(self, event_log):
        """
        Returns the length of the longest case in an event log.

        Parameters
        ----------
        event_log : list of dicts, where single dict represents a case
            The event log.

        Returns
        -------
        int :
            The length of the longest case in the event log.

        """

        max_case_length = 0
        for case in event_log:
            if case.__len__() > max_case_length:
                max_case_length = case.__len__()

        return max_case_length

    def get_indices_k_fold_validation(self, args, event_log):
        """
        Produces indices for each fold of a k-fold cross-validation.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        event_log : list of dicts, where single dict represents a case
            The event log.

        Returns
        -------
        training_indices_per_fold : list of arrays consisting of ints
            Indices of training cases from event log per fold.
        test_indices_per_fold : list of arrays consisting of ints
            Indices of test cases from event log per fold.

        """

        kFold = KFold(n_splits=args.num_folds, random_state=0, shuffle=False)

        train_indices_per_fold = []
        test_indices_per_fold = []

        for train_indices, test_indices in kFold.split(event_log):
            train_indices_per_fold.append(train_indices)
            test_indices_per_fold.append(test_indices)

        return train_indices_per_fold, test_indices_per_fold

    def get_indices_split_validation(self, args, event_log):
        """
        Produces indices for training and test set of a split-validation.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        event_log : list of dicts, where single dict represents a case
            The initial event log.

        Returns
        -------
        training_indices_per_fold : list of arrays consisting of ints
            Indices of training cases from event log.
        test_indices_per_fold : list of arrays consisting of ints
            Indices of test cases from event log.

        """

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test, random_state=0)

        train_indices_list = []
        test_indices_list = []

        for train_indices, test_indices in shuffle_split.split(event_log):
            train_indices_list.append(train_indices)
            test_indices_list.append(test_indices)

        return train_indices_list, test_indices_list

    def get_cases_of_fold(self, event_log, indices_per_fold):
        """
        Retrieves cases of a fold.

        Parameters
        ----------
        event_log : list of dicts, where single dict represents a case
            The initial event log.
        indices_per_fold : list of arrays consisting of ints
            Indices of either training or test cases from event log of current fold.

        Returns
        -------
        list of dicts : single dict represents a case
            Training or test cases from event log of current fold.

        """

        cases_of_fold = []

        for index in indices_per_fold[self.iteration_cross_validation]:
            cases_of_fold.append(event_log[index])

        return cases_of_fold

    def get_subsequences_of_cases(self, cases):
        """
        Creates subsequences of cases representing increasing prefix sizes.

        Parameters
        ----------
        cases : list of dicts, where single dict represents a case
            List of cases.

        Returns
        -------
        list of lists : each sublist contains a subsequence
            A subsequence is a subset of a case. Therefore, a subsequence contains one or multiple events, each
            represented by a dict.

        """

        subseq = []

        for case in cases:
            for idx_event in range(0, len(case._list)):
                # 0:i+1 -> get 0 up to n events of a process instance, since label is at t = n
                subseq.append(case._list[0:idx_event+1])


        return subseq

    def get_subsequence_of_case(self, case, prefix_size):
        """
        Crops a subsequence (= prefix) out of a whole case.

        Parameters
        ----------
        case : dict
            A case from the test set.
        prefix_size : int
            Size / Length of the subsequence to be created from a case.

        Returns
        -------
        list of dicts : single dict represents an event
            Subsequence / subset of a case whose length is prefix_size.

        """
        # 0 up to prefix-size; min prefix size = 1 with 2 elements
        return case._list[:prefix_size]

    def get_features_tensor(self, args, mode, event_log, subseq_cases):
        """
        Produces a vector-oriented representation of feature data as a 3-dimensional tensor.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        mode : str
            Indicates whether training or test features are passed.
        event_log : list of dicts, where single dict represents a case
            The encoded event log.
        subseq_cases : list of lists
            A sublist contains dicts, each of which represents a subset of a case.

        Returns
        -------
        ndarray : shape[S, T, F], S is number of samples, T is number of time steps, F is number of features.
            The features tensor.

        """

        num_features = self.get_num_features(args)
        max_case_length = self.get_max_case_length(event_log)

        if mode == 'train':
            features_tensor = numpy.zeros((len(subseq_cases),
                                           max_case_length,
                                           num_features), dtype=numpy.float64)
        else:
            features_tensor = numpy.zeros((1,
                                           max_case_length,
                                           num_features), dtype=numpy.float32)

        for idx_subseq, subseq in enumerate(subseq_cases):
            left_pad = max_case_length - len(subseq)

            for timestep, event in enumerate(subseq):

                # activity
                activity_values = event.get(args.activity_key)
                for idx, val in enumerate(activity_values):
                    features_tensor[idx_subseq, timestep + left_pad, idx] = val

                # time delta
                if args.include_time_delta:
                    features_tensor[idx_subseq, timestep + left_pad, self.get_length_of_activity_label()] = event[
                        args.time_delta_key]

                # context
                if self.context_exists():
                    start_idx = 0
                    if args.include_time_delta:
                        start_idx += 1

                    for attribute_idx, attribute_key in enumerate(self.context['attributes']):
                        attribute_values = event.get(attribute_key)

                        if not isinstance(attribute_values, list):
                            features_tensor[
                                idx_subseq, timestep + left_pad, start_idx + self.get_length_of_activity_label()] = attribute_values
                            start_idx += 1
                        else:
                            for idx, val in enumerate(attribute_values, start=start_idx):
                                features_tensor[
                                    idx_subseq, timestep + left_pad, idx + self.get_length_of_activity_label()] = val
                            start_idx += len(attribute_values)
        #
        return features_tensor

    def get_labels_tensor(self, args, cases_of_fold):
        """
        Produces a vector-oriented representation of labels as a 2-dimensional tensor.

        Parameters
        ----------
        args : Namespace
            Settings of the configuration parameters.
        cases_of_fold : list of dicts, where single dict represents a case
            Cases of the current fold.

        Returns
        -------
        list of lists :
            List of labels. Instead of a tuple, a label is represented by a (sub)list.

        """

        subseq_cases = self.get_subsequences_of_cases(cases_of_fold)
        num_classes = self.get_num_classes()
        class_labels = self.get_class_labels()

        labels_tensor = numpy.zeros((len(subseq_cases), num_classes), dtype=numpy.float64)

        for idx_subseq, subseq in enumerate(subseq_cases):
            for label in self.classes['labels']:
                if label == subseq[-1][args.outcome_key]:
                    labels_tensor[idx_subseq, class_labels.index(label)] = 1.0
                else:
                    labels_tensor[idx_subseq, class_labels.index(label)] = 0.0

        return labels_tensor

    def get_class_label(self, predictions):
        """
        Returns label of a predicted class.

        Parameters
        ----------
        predictions : list of floats
            A float represents the probability for a class.

        Returns
        -------
        tuple :
            The label of the predicted class.

        """
        labels = self.get_class_labels()
        max_prediction = 0
        class_id = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                label = labels[class_id]
            class_id += 1

        return label
