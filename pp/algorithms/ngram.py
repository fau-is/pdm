import pp.config as config
import pp.utils as utils
from pp.preprocessor import Preprocessor
import nltk
from nltk.util import ngrams
from statistics import mean
import operator

args = ""
process_instances = list()
n_grams_list = list()
accuracy_values = list()
accuracy_value = 0.0
accuracy_sum = 0.0

# define value of n for bigram, trigram, etc.
n = 6


# define function for ngrams
def get_ngrams(text, n):
    n_grams = ngrams(text, n)
    return [' '.join(grams) for grams in n_grams]


def train_ngrams(args, preprocess_manager):
    models = list()
    process_instances = list()

    # get train set
    for index, value in enumerate(
            preprocess_manager.train_index_per_fold[preprocess_manager.iteration_cross_validation]):
        process_instances.append(preprocess_manager.lines[value])

    # create loop: for each process instance create ngram
    n_grams_list = list()
    for process_instance in process_instances:
        n_grams = ngrams(process_instance, 1)
        n_grams_list.extend(n_grams)
    fdist = nltk.FreqDist(n_grams_list)
    most_freq = max(fdist.items(), key=operator.itemgetter(1))[0]
    for i in range(2, n + 1):
        n_grams_list = list()
        for process_instance in process_instances:
            # create ngrams and save in one list
            n_grams = ngrams(process_instance, i)
            n_grams_list.extend(n_grams)
        fdist = nltk.FreqDist(n_grams_list)
        model = dict()
        max_freq = dict()
        for k, v in fdist.items():
            target = k[i - 1]
            trace = k[0:i - 1]
            trace = ''.join(trace)
            if trace not in model:
                model[trace] = target
                max_freq[trace] = v
            else:
                if max_freq[trace] < v:
                    model[trace] = target
                    max_freq[trace] = v
        models.append(model)
    return models, most_freq


def test_ngrams(args, preprocess_manager, models, most_freq):
    process_instances = list()

    # get test set
    for index, value in enumerate(
            preprocess_manager.test_index_per_fold[preprocess_manager.iteration_cross_validation]):
        process_instances.append(preprocess_manager.lines[value])

    # prediction
    result = list()
    for process_instance in process_instances:
        pi = list(process_instance)
        trace_length = len(pi)
        actual_last_event = pi[trace_length - 1]
        prediction = most_freq
        for i in range(2, min(n + 1, trace_length)):
            trace = ''.join(pi[trace_length - i:trace_length - 1])
            if trace in models[i - 2]:
                prediction = models[i - 2][trace]
        if prediction == actual_last_event:
            result.append(1.0)
        else:
            result.append(0.0)

    return round(mean(result), 5)


if __name__ == '__main__':

    args = config.load()
    preprocessor = Preprocessor(args)

    if args.cross_validation:

        # iterate folds
        for iteration_cross_validation in range(0, args.num_folds):
            preprocessor.iteration_cross_validation = iteration_cross_validation
            models, most_freq = train_ngrams(args, preprocessor)

            args.iteration_cross_validation = iteration_cross_validation

            accuracy_value = test_ngrams(args, preprocessor, models, most_freq)
            accuracy_values.append(accuracy_value)

        # final output
        for index in range(0, len(accuracy_values)):
            utils.llprint("Accuracy of fold %i: %f\n" % (index + 1, accuracy_values[index]))
            accuracy_sum = accuracy_sum + accuracy_values[index]

        utils.llprint(
            "Average accuracy %i-fold cross-validation: %f\n" % (args.num_folds, accuracy_sum / args.num_folds))

    else:
        models, most_freq = train_ngrams(args, preprocessor)
        accuracy_value = test_ngrams(args, preprocessor, models, most_freq)

        # final output
        utils.llprint(
            "Accuracy %i/%i: %f\n" % (100 * (1 - 1 / args.num_folds), 100 * (1 / args.num_folds), accuracy_value))
        # iterate folds
