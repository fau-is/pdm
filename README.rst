##Next event prediction with deep learning 

# Overview

- Input: 
   - event log with the attributes "process instance", "even", "timestamp", "context attribute 1", "...", "context attribute n"
   - all categorial attributes (despite the "timestamp attriubte") are integer-mapped and the timestamp attribute has the format "dd.mm.yyyy-hh:mm:ss".
  
- Pre-processing variants:
   - "event" and categorical context attributes are encoded depending on the selected encoding technque  
   - numerical context attributes are min-max normalized
   - time delta attribute is min-max normalized 
  
- Encoding techniques:
   1. Paragraph embedding (doc2vec, shallow Neural Network)
   2. Ordinal encoding (integer-mapping)
   3. Binary encoding
   4. Hash encoding (proposed by Mehdiyev and Fettke (2017)) 
   5. Onehot encoding
    
- Deep learning architectures
   1. Multi Layer Perceptron (MLP) according to Theis et al. (2019)
   2. Long Short-Term Neural Network (LSTM) according to Tax et al. (2017)
   3. Convolutional Neural Network (CNN) according to Abdulrhman et al. (2019)
   4. Encoder
   
- Post processing: 
   - argmax


# Other useful repositories/sources
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




# Setup
The easiest way to setup an environment is to use Miniconda.

# Using Miniconda
   1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) 
   2. After setting up miniconda you can make use of the `conda` command in your command line (Powershell, CMD, Bash)
   3. To quickly install the `canepwdl` package, run `pip install -e .` inside the root directory.
   4. To install required libraries run 'pip install -r requirements.txt' inside the root directory.
   5. Install tensorflow (tensorflow.org/install)
   6. Now you can run the project.



