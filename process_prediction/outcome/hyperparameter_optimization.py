from sklearn.model_selection import ShuffleSplit

args = ''
max_case_length = ''
num_features = ''
num_classes = ''
iteration_cross_validation = ''

x_train = ''
y_train = ''
x_test = ''
y_test = ''


def create_data(arguments, event_log, preprocessor, cases_of_fold):
    """
    Generates data to train and test/evaluate a model during hyperparameter optimization (hpo) with Optuna.

    Parameters
    ----------
    arguments : Namespace
        Settings of the configuration parameters.
    event_log : list of dicts, where single dict represents a case
        pm4py.objects.log.log.EventLog object representing an event log.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    cases_of_fold : list of dicts, where single dict represents a case
        Cases of the current fold.

    Returns
    -------
    x_train : ndarray, shape[S, T, F], S is number of samples, T is number of time steps, F is number of features.
        The features of the training set.
    y_train : ndarray, shape[S, T], S is number of samples, T is number of time steps.
        The labels of the training set.
    x_test : ndarray, shape[S, T, F], S is number of samples, T is number of time steps, F is number of features.
        The features of the test set.
    y_test : ndarray, shape[S, T], S is number of samples, T is number of time steps.
        The labels of the test set.
    args : Namespace
        Settings of the configuration parameters.

    """

    global args
    global max_case_length
    global num_features
    global num_classes
    global iteration_cross_validation

    global x_train
    global y_train
    global x_test
    global y_test

    # init parameters
    args = arguments
    max_case_length = preprocessor.get_max_case_length(event_log)
    num_features = preprocessor.get_num_features(args)
    num_classes = preprocessor.get_num_outcome_classes()
    iteration_cross_validation = preprocessor.iteration_cross_validation

    # preprocess data
    train_indices, test_indices = train_test_split_for_hyperparameter_optimization(cases_of_fold)
    train_cases, test_cases = retrieve_train_test_instances(cases_of_fold, train_indices, test_indices)
    train_subseq_cases, test_subseq_cases = retrieve_train_test_subsequences(train_cases, test_cases, preprocessor)

    x_train = preprocessor.get_features_tensor(args, 'train', event_log, train_subseq_cases)
    y_train = preprocessor.get_labels_tensor(args, train_subseq_cases)

    x_test = preprocessor.get_features_tensor(args, 'train', event_log, test_subseq_cases)
    y_test = preprocessor.get_labels_tensor(args, test_subseq_cases)


def train_test_split_for_hyperparameter_optimization(cases):
    """
    Executes a split-validation and retrieves indices of training and test cases for hpo.

    Parameters
    ----------
    cases : list of dicts, where single dict represents a case
        Cases of the training set.

    Returns
    -------
    hpo_train_indices[0] : list of ints
        Indices of training cases for hpo.
    hpo_test_indices[0] : list of ints
        Indices of test cases for hpo.

    """

    shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test_hpo, random_state=0)

    hpo_train_indices = []
    hpo_test_indices = []

    for train_indices, test_indices in shuffle_split.split(cases):
        hpo_train_indices.append(train_indices)
        hpo_test_indices.append(test_indices)

    return hpo_train_indices[0], hpo_test_indices[0]


def retrieve_train_test_instances(cases, train_indices, test_indices):
    """
    Retrieves training and test cases from indices for hpo.

    Parameters
    ----------
    cases : list of dicts, where single dict represents a case
        Cases of the training set of the event log.
    train_indices : list of ints
        Indices of training cases for hpo.
    test_indices : list of ints
        Indices of test cases for hpo.

    Returns
    -------
    train_cases : list of dicts, where single dict represents a case
        Training cases for hpo.
    test_cases : list of dicts, where single dict represents a case
        Test cases for hpo.

    """

    train_cases = []
    test_cases = []

    for idx in train_indices:
        train_cases.append(cases[idx])

    for idx in test_indices:
        test_cases.append(cases[idx])

    return train_cases, test_cases


def retrieve_train_test_subsequences(train_cases, test_cases, preprocessor):
    """
    Creates subsequences of training and test cases for hpo. Subsequences represent subsets of a case with increasing
    length / prefix sizes.

    Parameters
    ----------
    train_cases : list of dicts, where single dict represents a case
        Training cases for hpo.
    test_cases : list of dicts, where single dict represents a case
        Test cases for hpo.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    train_subseq_cases : list of dicts, where single dict represents a subsequence of a case
        Subsequences of the training cases for hpo.
    test_subseq_cases : list of dicts, where single dict represents a subsequence of a case
        Subsequences of the training cases for hpo.

    """

    train_subseq_cases = preprocessor.get_subsequences_of_cases(train_cases)
    test_subseq_cases = preprocessor.get_subsequences_of_cases(test_cases)

    return train_subseq_cases, test_subseq_cases