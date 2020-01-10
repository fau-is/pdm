is-pcm


This project is a prototype of a predictive compliance monitoring technique. The technique consists of the three components: (1) label generation by log replay (compliance checking), (2) label prediction with outcome-oriented predictive process monitoring (LSTM), and explainable predictions by using a layer-wise relevance probagation (lrp) backward pass.

Note our implementation of the lrp, esp. for process predictions, based on the code provided by the paper "Explaining Recurrent Neural Network Predictions in Sentiment Analysis" from Arras et al. (2017). 

There are two conceptual versions of the project: One considers additional context attributes (besides the outcome attribute to be predicted). This version can be found on the branch "dev-context". The other version is not context-sensitive.


Other useful repositories/sources

- https://github.com/verenich/ProcessSequencePrediction (CAiSE 2017)

- https://github.com/AdaptiveBProcess/GenerativeLSTM/tree/master/models (BPM 2019) 

- https://github.com/tnolle/binet (4 Papers)

- https://github.com/ProminentLab/DREAM-NAP

- http://contrib.scikit-learn.org/categorical-encoding/

- https://github.com/delas/plg (synthetic eventlog generator)

- https://github.com/keras-team/keras-contrib (keras extension)

- https://github.com/irhete/predictive-monitoring-thesis (hyperparameter optimization for lstms)

- https://docs.python-guide.org/writing/structure/ (structure of python project)

- https://keras.io/examples/babi_memnn/ (another DNC implementation that could be more efficient) 

Setup
The easiest way to setup an environment is to use Miniconda.

Using Miniconda

1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) 

2. After setting up miniconda you can make use of the `conda` command in your command line (Powershell, CMD, Bash)

3. To quickly install the `canepwdl` package, run `pip install -e .` inside the root directory.

4. To install required libraries run 'pip install -r requirements.txt' inside the root directory.

5. Install tensorflow (tensorflow.org/install)

6. Now you can run the project.



