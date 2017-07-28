# %load "SentimentAnalysisWithCntk.py"
import os
import cntk as C
import csv
import copy
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
import itertools
from nltk.corpus import stopwords


CSV_SENTIMENT_VALUE_INDEX = 0
CSV_SENTIMENT_TEXT_INDEX = 5
SENTIMENT_VALUE_NEGATIVE = 0
SENTIMENT_VALUE_NEUTRAL = 2
SENTIMENT_VALUE_POSITIVE = 4
 
def process_str(string):
    string = string.strip().lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string
 
# Creates the reader
def create_reader(path, is_training, input_dim, label_dim):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features = C.io.StreamDef(field='x', shape=input_dim,   is_sparse=True),
        labels   = C.io.StreamDef(field='y', shape=label_dim,   is_sparse=False)
    )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)
 
 
# Defines the LSTM model for classifying sequences
def lstm_sequence_classifier(features, num_classes, embedding_dim, LSTM_dim):
    classifier = C.layers.Sequential([C.layers.Embedding(embedding_dim),
                                      C.layers.Recurrence(C.layers.LSTM(LSTM_dim)),
                                      C.sequence.last,
                                      C.layers.Dense(num_classes)])
    return classifier(features)
 

# Creates and trains a LSTM sequence classification model
def train_sequence_classifier():
    input_dim = 8000
    hidden_dim = 25
    embedding_dim = 50
    num_classes = 3
    X_data = []
    Y_data = []
    invalid_row_count = 0
    negative_class_vec = [1 , 0 , 0]
    neutral_class_vec  = [0 , 1 , 0]
    positive_class_vec = [0 , 0 , 1]
    max_document_length = 0
    twitter_data_file = "C:/dsvm/notebooks/CNTK-Samples-2-0/Examples/SequenceClassification/SimpleExample/Python/tf_dataset_small.csv"
    print ("Started reading training dataset")
    with open(twitter_data_file, newline='', encoding="ISO-8859-1") as f:
        twitter_data_cvsreader = csv.reader(f)
        for row in twitter_data_cvsreader:
            sentiment_value = int(row[CSV_SENTIMENT_VALUE_INDEX])
            sentment_text = process_str(row[CSV_SENTIMENT_TEXT_INDEX])
            max_document_length = max(max_document_length, len(sentment_text.split(" ")))
            if sentiment_value == SENTIMENT_VALUE_NEGATIVE:
                X_data.append(sentment_text)
                Y_data.append(negative_class_vec)
            elif sentiment_value == SENTIMENT_VALUE_NEUTRAL:
                X_data.append(sentment_text)
                Y_data.append(neutral_class_vec)
            elif sentiment_value == SENTIMENT_VALUE_POSITIVE:
                X_data.append(sentment_text)
                Y_data.append(positive_class_vec)
            else:
                invalid_row_count += 1
            #Invalid value. do not consider this row data
    f.close()
   
    # Convert list to np format for analysis
    Y_np_data = np.array(Y_data)
    tokens_docs = [doc.split(" ") for doc in X_data]
    all_tokens = itertools.chain.from_iterable(tokens_docs)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
    token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokens_docs]
    
    for rows in range(0,len(token_ids)):
        if len(token_ids[rows])< max_document_length:
            token_ids[rows] = np.pad(token_ids[rows], (0, max_document_length - len(token_ids[rows])%max_document_length), 'constant').tolist()
    
    x_newdata = np.asarray(token_ids)
    vec = OneHotEncoder()
    X = vec.fit_transform(x_newdata)
    X = X.tocoo()

    with open("C:/dsvm/ashima.ctf",'w') as f:
        
        for a,b,c in zip(X.row, X.col, X.data):
            p = a+1
            break
        
        
        for i,j,v in zip(X.row, X.col, X.data):
            output = ""
            output = str(i) + " |x " + str(j) + ":" + str(v)
            if p != i:
                output = output + " |y " + str(Y_np_data[i][0]) + " " + str(Y_np_data[i][1]) + " " + str(Y_np_data[i][2])
            p = i
            f.write(output + '\n')
            
    f.close() 
    
    print ("Finished reading training dataset")
    if invalid_row_count > 0:
        print ("Invalid Row Count : " + str(invalid_row_count))
    print ("X_Y_Data Length : " + str(data_size))
    # Input variables denoting the features and label data
    features = C.sequence.input_variable(shape=input_dim, is_sparse=True)
    label = C.input_variable(num_classes)
 
    # Instantiate the sequence classification model
    classifier_output = lstm_sequence_classifier(features, num_classes, embedding_dim, hidden_dim)
 
    ce = C.cross_entropy_with_softmax(classifier_output, label)
    pe = C.classification_error(classifier_output, label)
 
    rel_path = r"../../../../Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    print(path1)
    path = "C:/dsvm/ashima.ctf"
 
    reader = create_reader(path, True, input_dim, num_classes)
 
    input_map = {
        features : reader.streams.features,
        label    : reader.streams.labels
    }
 
    lr_per_sample = C.learning_rate_schedule(0.1, C.UnitType.sample)
 
    # Instantiate the trainer object to drive the model training
    progress_printer = C.logging.ProgressPrinter(0)
    trainer = C.Trainer(classifier_output, (ce, pe),
                        C.sgd(classifier_output.parameters, lr=lr_per_sample),
                        progress_printer)
 
    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200
 
    for i in range(251):
        mb = reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)
 
    evaluation_average = copy.copy(trainer.previous_minibatch_evaluation_average)
    loss_average = copy.copy(trainer.previous_minibatch_loss_average)
 
    F = C.ops.functions.Function.save(classifier_output, "C:/dsvm/ashimahack.dnn")

    return evaluation_average, loss_average
 
if __name__ == '__main__':
    
 
  
 
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # C.try_set_default_device(C.cpu())
 
    error, _ = train_sequence_classifier()
    print("Error: %f" % error)