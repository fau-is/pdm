from __future__ import division
import csv
import numpy
import copy
import process_prediction.utils as utils
from sklearn.model_selection import KFold, ShuffleSplit
import gensim


class Preprocessor(object):
    data_structure = {
        'support': {
            'num_folds': 1,
            'data_dir': "",
            'ascii_offset': 161,
            'data_format': "%d.%m.%Y-%H:%M:%S",
            'train_index_per_fold': [],
            'test_index_per_fold': [],
            'iteration_cross_validation': 0,
            'elements_per_fold': 0,
            'event_labels': [],
            'event_types': [],
            'map_event_label_to_event_id': [],
            'map_event_id_to_event_label': [],
            'map_event_type_to_event_id': [],
            'map_event_id_to_event_type': [],
            'end_process_instance': '!',
            'embedding_model': ""
        },

        'meta': {
            'num_features': 0,
            'num_event_ids': 0,
            'max_length_process_instance': 0,
            'num_attributes_context': 0,
            'num_attributes_control_flow': 3,  # process instance id, event id and timestamp
            'num_process_instances': 0
        },

        'data': {
            'process_instances': [],
            'ids_process_instances': [],
            'context_attributes_process_instances': [],

            'train': {
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            },

            'test': {
                'process_instances': [],
                'context_attributes': [],
                'event_ids': []
            }
        }
    }

    def __init__(self, args):

        utils.llprint("Initialization ... \n")
        self.data_structure['support']['num_folds'] = args.num_folds
        self.data_structure['support']['data_dir'] = args.data_dir + args.data_set
        self.get_sequences_from_eventlog()
        self.data_structure['support']['elements_per_fold'] = \
            int(round(
                self.data_structure['meta']['num_process_instances'] / self.data_structure['support']['num_folds']))
        self.data_structure['meta']['num_features'] = args.embedding_dim

        # add end marker of process instance
        self.data_structure['data']['process_instances'] = list(map(lambda x: x + ['!'], self.data_structure['data']['process_instances']))
        self.data_structure['meta']['max_length_process_instance'] = max(map(lambda x: len(x), self.data_structure['data']['process_instances']))

        # structures for predicting next activities
        self.data_structure['support']['event_labels'] = list(
            map(lambda x: set(x), self.data_structure['data']['process_instances']))
        self.data_structure['support']['event_labels'] = list(
            set().union(*self.data_structure['support']['event_labels']))
        self.data_structure['support']['event_labels'].sort()
        self.data_structure['support']['event_types'] = copy.copy(self.data_structure['support']['event_labels'])
        self.data_structure['support']['map_event_label_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_id_to_event_label'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_type_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_types']))
        self.data_structure['support']['map_event_id_to_event_type'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_types']))
        self.data_structure['meta']['num_event_ids'] = len(self.data_structure['support']['event_labels'])

        if not args.explain:
            utils.llprint("Create embedding model ... \n")
            self.data_structure['support']['embedding_model'] = self.create_embedding_model(args)
        else:
            utils.llprint("Load embedding model ... \n")
            self.data_structure['support']['embedding_model'] = gensim.models.Word2Vec.load('%s%sembeddings.model' % ("nextevent", args.model_dir[1:]))

        if args.cross_validation:
            self.set_indices_k_fold_validation()
        else:
            self.set_indices_split_validation(args)

    def get_sequences_from_eventlog(self):

        id_latest_process_instance = ''
        process_instance = []
        first_event_of_process_instance = True
        output = True

        file = open(self.data_structure['support']['data_dir'], 'r')
        reader = csv.reader(file, delimiter=';', quotechar='|')
        next(reader, None)

        for event in reader:

            id_current_process_instance = event[0]

            if output:
                output = False

            if id_current_process_instance != id_latest_process_instance:
                self.add_data_to_data_structure(id_current_process_instance, 'ids_process_instances')
                id_latest_process_instance = id_current_process_instance

                if not first_event_of_process_instance:
                    self.add_data_to_data_structure(process_instance, 'process_instances')

                process_instance = []

                self.data_structure['meta']['num_process_instances'] += 1

            process_instance.append(event[1])
            first_event_of_process_instance = False

        file.close()

        self.add_data_to_data_structure(process_instance, 'process_instances')

        self.data_structure['meta']['num_process_instances'] += 1


    def create_embedding_model(self, args):
        file = open(self.data_structure['support']['data_dir'], 'r')
        reader = csv.reader(file, delimiter=';', quotechar='|')
        next(reader, None)
        data_set = list()
        embedding_dim = args.embedding_dim
        epochs = args.embedding_epochs

        # create data set
        for row in reader:
            data_set.append([row[1]])

        file.close()

        # train model
        # note each word is handled as a sentence
        model = gensim.models.Word2Vec(data_set, size=embedding_dim, window=3, min_count=1)

        for epoch in range(epochs):
            if epoch % 2 == 0:
                print('Now training epoch %s' % epoch)
            model.train(data_set, total_examples=len(data_set), epochs=epochs)
            model.alpha -= 0.002  # decrease learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        # save
        model.save('./%s%sembeddings.model' % (args.task, args.model_dir[1:]), sep_limit=2000000000)

        # print(model.wv.most_similar(positive="Take"))
        # print(0)

        return model

    def set_training_set(self):

        utils.llprint("Get training instances ... \n")
        process_instances_train, context_attributes_train, _ = \
            self.get_instances_of_fold('train')

        utils.llprint("Create cropped training instances ... \n")
        cropped_process_instances, cropped_context_attributes, next_events = \
            self.get_cropped_instances(
                process_instances_train,
                context_attributes_train)

        utils.llprint("Create training set data as tensor ... \n")
        features_data = self.get_data_tensor(cropped_process_instances,
                                             'train')

        utils.llprint("Create training set label as tensor ... \n")
        labels = self.get_label_tensor(cropped_process_instances,
                                       next_events)

        self.data_structure['data']['train']['features_data'] = features_data
        self.data_structure['data']['train']['labels'] = labels

    def get_event_type(self, predictions):
        max_prediction = 0
        event_type = ''
        index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                event_type = self.data_structure['support']['map_event_id_to_event_type'][index]
            index += 1

        return event_type

    """
    def check_for_context_attributes(self, event):
        if len(event) == self.data_structure['meta']['num_attributes_control_flow']:
            utils.llprint("No context attributes found ...\n")
        else:
            self.data_structure['meta']['num_attributes_context'] = len(event) - self.data_structure['meta'][
                'num_attributes_control_flow']
            utils.llprint("%d context attribute(s) found ...\n" % self.data_structure['meta']['num_attributes_context'])
    """

    def add_data_to_data_structure(self, values, structure):
        self.data_structure['data'][structure].append(values)

    def get_context_attributes_of_event(self, event):
        context_attributes_event = []
        for attribute_index in range(self.data_structure['meta']['num_attributes_control_flow'],
                                     self.data_structure['meta']['num_attributes_control_flow'] +
                                     self.data_structure['meta']['num_attributes_context']):
            context_attributes_event.append(event[attribute_index])

        return context_attributes_event

    def add_event_to_process_instance(self, event, process_instance):
        return process_instance + chr(int(event[1]) + self.data_structure['support']['ascii_offset'])

    '''
    Produces indicies for each fold of a k-fold cross-validation.
    '''

    def set_indices_k_fold_validation(self):

        kFold = KFold(n_splits=self.data_structure['support']['num_folds'], random_state=0, shuffle=False)

        for train_indices, test_indices in kFold.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    '''
    Produces indices for train and test set of a split-validation.
    '''

    def set_indices_split_validation(self, args):

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test, random_state=0)

        for train_indices, test_indices in shuffle_split.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    '''
    Retrieves instances of a fold.
    '''

    def get_instances_of_fold(self, mode):

        process_instances_of_fold = []
        context_attributes_of_fold = []
        event_ids_of_fold = []

        for index, value in enumerate(self.data_structure['support'][mode + '_index_per_fold'][
                                          self.data_structure['support']['iteration_cross_validation']]):
            process_instances_of_fold.append(self.data_structure['data']['process_instances'][value])
            event_ids_of_fold.append(self.data_structure['data']['ids_process_instances'][value])

            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_of_fold.append(
                    self.data_structure['data']['context_attributes_process_instances'][value])

        if mode == 'test':
            self.data_structure['data']['test']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['test']['context_attributes'] = context_attributes_of_fold
            self.data_structure['data']['test']['event_ids'] = event_ids_of_fold
            return

        return process_instances_of_fold, context_attributes_of_fold, event_ids_of_fold

    '''
    Crops prefixes out of instances.
    '''

    def get_cropped_instances(self, process_instances, context_attributes_process_instances):

        cropped_process_instances = []
        cropped_context_attributes = []
        next_events = []

        if self.data_structure['meta']['num_attributes_context'] > 0:

            for process_instance, context_attributes_process_instance in zip(process_instances,
                                                                             context_attributes_process_instances):
                for i in range(0, len(process_instance)):

                    if i == 0:
                        continue

                    # 0:i -> get 0 up to n-1 events of a process instance, since n is the label 
                    cropped_process_instances.append(process_instance[0:i])
                    cropped_context_attributes.append(context_attributes_process_instance[0:i])
                    # label
                    next_events.append(process_instance[i])
        else:
            for process_instance in process_instances:
                for i in range(0, len(process_instance)):

                    if i == 0:
                        continue
                    cropped_process_instances.append(process_instance[0:i])
                    next_events.append(process_instance[i])

        return cropped_process_instances, cropped_context_attributes, next_events

    def get_cropped_instance_label(self, prefix_size, process_instance):
        """
        Crops next activity label out of a single process instance.
        """

        if prefix_size == len(process_instance) - 1:
            # end marker
            return self.data_structure["support"]["end_process_instance"]
        else:
            return process_instance[prefix_size]  # label of next act




    def get_cropped_instance(self, prefix_size, process_instance):
        """
        Crops prefixes out of a single process instance.
        """

        cropped_process_instance = process_instance[:prefix_size]

        return cropped_process_instance

    '''
    Produces a vector-oriented representation of data as 3-dimensional tensor.
    '''

    def get_data_tensor(self, cropped_process_instances, mode):

        if mode == 'train':
            data_set = numpy.zeros((
                len(cropped_process_instances),
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.float64)
        else:
            data_set = numpy.zeros((
                1,
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.float32)


        model = self.data_structure["support"]["embedding_model"]
        for index, cropped_process_instance in enumerate(cropped_process_instances):
            for index_, activity in enumerate(cropped_process_instance):

                # apply embeddings
                # print(model.wv.vocab)

                data_set[index, index_, :] = model.wv[activity]

        return data_set

    def get_data_tensor_for_single_prediction(self, cropped_process_instance):

        data_set = self.get_data_tensor(
            [cropped_process_instance],
            'test')

        return data_set

    '''
    Produces a vector-oriented representation of label as 2-dimensional tensor.
    '''

    def get_label_tensor(self, cropped_process_instances, next_events):

        label = numpy.zeros((len(cropped_process_instances), len(self.data_structure['support']['event_types'])),
                            dtype=numpy.float64)

        for index, cropped_process_instance in enumerate(cropped_process_instances):

            for event_type in self.data_structure['support']['event_types']:

                if event_type == next_events[index]:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 1
                else:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 0

        return label
