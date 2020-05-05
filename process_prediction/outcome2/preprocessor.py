from __future__ import division
import csv
import numpy
import copy
import gensim
import process_prediction.utils as utils
from sklearn.model_selection import KFold, ShuffleSplit


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
            'embedding_model': ""

        },

        'meta': {
            'num_features': 0,
            'num_event_ids': 0,
            'max_length_process_instance': 0,
            'num_process_instances': 0,
            'num_classes': 0,  # i.g., different values of the compliance attribute
            'val_classes': [],
            'map_class_val_id': [],
            'map_class_id_val': []
        },

        'data': {
            'process_instances': [],
            'labels': [],
            'ids_process_instances': [],
            'context_attributes_process_instances': [],

            'train': {
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            },

            'test': {
                'process_instances': [],
                'event_ids': [],
                'labels': numpy.ndarray([])
            },

            'all': {
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            }

        }
    }

    def __init__(self, args, meta):

        utils.llprint("Initialization ... \n")
        self.data_structure['support']['num_folds'] = args.num_folds
        if meta:
            self.data_structure['support']['data_dir'] = args.data_dir + args.data_set
        else:
            self.data_structure['support']['data_dir'] = args.data_dir + args.data_set_out2

        self.get_sequences_from_event_log()
        self.data_structure['support']['elements_per_fold'] = \
            int(round(
                self.data_structure['meta']['num_process_instances'] / self.data_structure['support']['num_folds']))
        self.data_structure['meta']['num_features'] = args.embedding_dim  # dimension of embeddings
        self.data_structure['meta']['max_length_process_instance'] = max(
            map(lambda x: len(x), self.data_structure['data']['process_instances']))

        self.data_structure['meta']['num_classes'], \
        self.data_structure['meta']['val_classes'], \
        self.data_structure['meta']['map_class_val_id'], \
        self.data_structure['meta']['map_class_id_val'], \
            = self.get_classes()

        if not args.explain:
            utils.llprint("Create embedding model ... \n")
            self.data_structure['support']['embedding_model'] = self.create_embedding_model(args)
        else:
            utils.llprint("Load embedding model ... \n")
            self.data_structure['support']['embedding_model'] = gensim.models.Word2Vec.load('%s%sembeddings.model' % ("outcome2", args.model_dir[1:]))

        if args.cross_validation:
            self.set_indices_k_fold_validation()
        else:
            self.set_indices_split_validation(args)

    def get_classes(self):
        file = open(self.data_structure['support']['data_dir'], 'r')
        reader = csv.reader(file, delimiter=';', quotechar='|')
        next(reader, None)
        labels = list()
        map_class_val_id = dict()
        map_class_id_val = dict()

        for row in reader:
            labels.append(row[3])

        file.close()

        for index, value in enumerate(list(set(labels))):
            map_class_val_id[value] = index
            map_class_id_val[index] = value

        return len(list(set(labels))), list(set(labels)), map_class_val_id, map_class_id_val

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
        model = gensim.models.Word2Vec(data_set, alpha=0.025, size=embedding_dim, window=5, min_count=1)

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

    def get_sequences_from_event_log(self):

        id_latest_process_instance = ''
        first_event_of_process_instance = True
        output = True
        process_instance = []
        process_instance_labels = []
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
                    self.add_data_to_data_structure(process_instance_labels, 'labels')

                process_instance = []
                process_instance_labels = []

                self.data_structure['meta']['num_process_instances'] += 1

            process_instance.append(event[1])
            process_instance_labels.append(event[3])
            first_event_of_process_instance = False

        file.close()

        self.add_data_to_data_structure(process_instance, 'process_instances')
        self.add_data_to_data_structure(process_instance_labels, 'labels')

        self.data_structure['meta']['num_process_instances'] += 1

    def set_training_set(self):

        utils.llprint("Get training instances ... \n")
        process_instances_train, labels_train, _ = \
            self.get_instances_of_fold('train')

        utils.llprint("Create training set data as tensor ... \n")
        features_data = self.get_data_tensor(process_instances_train, 'train')

        utils.llprint("Create training set label as tensor ... \n")
        labels = self.get_label_tensor(process_instances_train,
                                       labels_train)

        self.data_structure['data']['train']['features_data'] = features_data
        self.data_structure['data']['train']['labels'] = labels

    def get_class_val(self, predictions):
        max_prediction = 0
        val_class = ''
        index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                val_class = self.data_structure['meta']['map_class_id_val'][index]
            index += 1

        return val_class


    def add_data_to_data_structure(self, values, structure):
        self.data_structure['data'][structure].append(values)


    def set_indices_k_fold_validation(self):
        """
        Produces indices for each fold of a k-fold cross-validation.
        """

        kFold = KFold(n_splits=self.data_structure['support']['num_folds'], random_state=0, shuffle=True)

        for train_indices, test_indices in kFold.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)


    def set_indices_split_validation(self, args):
        """
        Produces indices for train and test set of a split-validation.
        """

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test, random_state=0)

        for train_indices, test_indices in shuffle_split.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)


    def get_instances_of_fold(self, mode):
        """
        Retrieves instances of a fold.
        """

        process_instances_of_fold = []
        labels_of_fold = []
        process_instances_ids_of_fold = []

        for index, value in enumerate(self.data_structure['support'][mode + '_index_per_fold'][
                                          self.data_structure['support']['iteration_cross_validation']]):
            process_instances_of_fold.append(self.data_structure['data']['process_instances'][value])
            process_instances_ids_of_fold.append(self.data_structure['data']['ids_process_instances'][value])
            labels_of_fold.append(self.data_structure['data']['labels'][value])

        labels_of_fold = [labels[-1] for labels in labels_of_fold]

        if mode == 'test':
            self.data_structure['data']['test']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['test']['labels'] = [labels_instance[-1] for labels_instance in labels_of_fold]
            self.data_structure['data']['test']['event_ids'] = process_instances_ids_of_fold
            return

        return process_instances_of_fold, labels_of_fold, process_instances_ids_of_fold



    def get_data_tensor(self, process_instances, mode):
        """
        Produces a vector-oriented representation of data as 3-dimensional tensor.
        """

        # create structure
        if mode == 'train':
            data_set = numpy.zeros((
                len(process_instances),
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.float64)
        else:
            data_set = numpy.zeros((
                1,
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.float32)

        # fill structure
        model = self.data_structure["support"]["embedding_model"]
        for index, cropped_process_instance in enumerate(process_instances):
            for index_, activity in enumerate(cropped_process_instance):

                # apply embeddings
                # print(model.wv.vocab)
                data_set[index, index_, :] = model.wv[activity]

        return data_set

    def get_data_tensor_for_single_prediction(self, process_instance):

        data_set = self.get_data_tensor(
            [process_instance],
            'test')

        return data_set


    def get_label_tensor(self, process_instances, labels_):
        """
        Produces a vector-oriented representation of label as 2-dimensional tensor.
        """

        # create data structure
        labels = numpy.zeros((len(process_instances), self.data_structure['meta']['num_classes']),
                             dtype=numpy.int_)

        # fill data structure
        for index, cropped_process_instance in enumerate(process_instances):

            # one hot encoding
            for val_class in self.data_structure['meta']['val_classes']:

                if val_class == labels_[index]:
                    labels[index, self.data_structure['meta']['map_class_val_id'][val_class]] = 1
                else:
                    labels[index, self.data_structure['meta']['map_class_val_id'][val_class]] = 0

        return labels

    def get_process_instance(self):
        return self.data_structure['data']['process_instances'], self.data_structure['data']['labels']
