from source.LSTM.LSTM_bidi import *
from source.util.heatmap import html_heatmap
import codecs
import numpy as np
from IPython.display import display, HTML

if __name__ == "__main__":

    def get_test_sentence(sent_idx):
        """Returns an SST test set sentence and its true label, sent_idx must be an integer in [1, 2210]"""
        idx = 1
        with codecs.open("./data/sequence_test.txt", 'r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip('\n')
                line = line.split('\t')
                true_class = int(line[0]) - 1  # true class (construct label)
                words = line[1].split(' | ')  # sentence as list of words
                if idx == sent_idx:
                    return words, true_class
                idx += 1


    def predict(words):
        """Returns the classifier's predicted class"""
        net = LSTM_bidi()  # load trained LSTM model
        w_indices = [net.voc.index(w) for w in words]  # convert input sentence to word IDs
        net.set_input(w_indices)  # set LSTM input sequence
        scores = net.forward()  # classification prediction scores
        return np.argmax(scores)

    ## get test set
    words, _ = get_test_sentence(291)  # get test set
    words = [str(w).replace("\r", "") for w in words]

    # Alternatively, uncomment one of the following sentences, or define your own sequence (only words contained in the vocabulary are supported!)
    # words = ['this','movie','was','actually','neither','that','funny',',','nor','super','witty','.']

    # get predicted class
    predicted_class = predict(words)
    # define relevance target class
    target_class = predicted_class

    print(words)
    print("\npredicted class:", predicted_class)

    # compute lrp relevances ####################################
    # LRP hyperparameters:
    eps = 0.001  # small positive number
    bias_factor = 0.0  # recommended value

    net = LSTM_bidi()  # load trained LSTM model

    w_indices = [net.voc.index(w) for w in words]  # convert input sentence to word IDs
    Rx, Rx_rev, R_rest = net.lrp(w_indices, target_class, eps, bias_factor)  # perform LRP
    R_words = np.sum(Rx + Rx_rev, axis=1)  # compute word-level LRP relevances

    scores = net.s.copy()  # classification prediction scores

    print("prediction scores:", scores)
    print("\nLRP target class:", target_class)
    print("\nLRP relevances:")
    for idx, w in enumerate(words):
        print("\t\t\t" + "{:8.2f}".format(R_words[idx]) + "\t" + w)
    print("\nLRP heatmap:")
    display(HTML(html_heatmap(words, R_words)))

    # How to sanity check global relevance conservation:
    bias_factor = 1.0  # value to use for sanity check
    Rx, Rx_rev, R_rest = net.lrp(w_indices, target_class, eps, bias_factor)
    R_tot = Rx.sum() + Rx_rev.sum() + R_rest.sum()  # sum of all "input" relevances

    print(R_tot)
    print("Sanity check passed? ", np.allclose(R_tot, net.s[target_class]))

    """
    # compute sa/gi relevances #####################################
    net = LSTM_bidi()  # load trained LSTM model

    w_indices = [net.voc.index(w) for w in words]  # convert input sentence to word IDs
    Gx, Gx_rev = net.backward(w_indices, target_class)  # perform gradient backpropagation
    R_words_SA = (np.linalg.norm(Gx + Gx_rev, ord=2, axis=1)) ** 2  # compute word-level Sensitivity Analysis relevances
    R_words_GI = ((Gx + Gx_rev) * net.x).sum(axis=1)  # compute word-level GradientxInput relevances

    scores = net.s.copy()  # classification prediction scores

    print("prediction scores:       ", scores)
    print("\nSA/GI target class:      ", target_class)
    print("\nSA relevances:")
    for idx, w in enumerate(words):
        print("\t\t\t" + "{:8.2f}".format(R_words_SA[idx]) + "\t" + w)
    print("\nSA heatmap:")
    display(HTML(html_heatmap(words, R_words_SA)))
    print("\nGI relevances:")
    for idx, w in enumerate(words):
        print("\t\t\t" + "{:8.2f}".format(R_words_GI[idx]) + "\t" + w)
    print("\nGI heatmap:")
    display(HTML(html_heatmap(words, R_words_GI)))
    """


