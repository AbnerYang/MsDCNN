# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(0)
import pandas as pd
import cPickle
import os
import time
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D, Convolution1D, Convolution2D
from keras.layers.advanced_activations import PReLU
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import SGD

import gc
gc.collect()

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

data_sources = [
  "ag_news_csv",
#     "amazon_review_polarity_csv",
#     "amazon_review_full_csv",
#      "sogou_news_csv",
#     "yahoo_answers_csv",
#     "yelp_review_full_csv",
#     "yelp_review_polarity_csv"
#     "dbpedia_csv"
]
embedding_dim = 300
batch_size = 256
num_epochs = 10
mode = 'word' # word
vocab_size = 30000


def DenseBlock(x1, kernel, filterNum, layerNum = 3):
    x_base = [x1]
    for i in range(layerNum):
        x = Convolution1D(filterNum, kernel[i], padding='same')(x1)
        #x = Dropout(0.5)(x)
        x = PReLU()(x)
        #x = Activation('relu')(x)
        x_base.append(x)
        x1 = concatenate(x_base)
    return x1

def MsDenselyCNN(inputs_x, vocab_size, sequence_length, embedding_dim, embedding_matrix=None):
    if embedding_matrix is None:
        print("Training without pre-train embedding.")
        x = Embedding(vocab_size, embedding_dim, input_length=sequence_length, trainable=True)(inputs_x)
    else:
        print("Training with pre-train embedding.")
        x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=False)(inputs_x)
    x = DenseBlock(x, [1, 2, 3, 4], 128, layerNum=4)
    x = MaxPooling1D(2)(x)
#     x = BatchNormalization()(x)
    x = DenseBlock(x, [1, 2, 3, 4], 128, layerNum=4)
#     x = MaxPooling1D(2)(x)
#     x = DenseBlock(x, [1, 2, 3, 4], 128, layerNum=4)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.25)(x)
#     x = BatchNormalization()(x)
    
#     x = Dense(128)(x)
#     x = Activation('relu')(x)
#     x = BatchNormalization()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    return x

def CNN(inputs_x, vocab_size, sequence_length, embedding_dim, embedding_matrix=None):
    if embedding_matrix is None:
        print("Training without pre-train embedding.")
        x = Embedding(vocab_size, embedding_dim, input_length=sequence_length, trainable=True)(inputs_x)
    else:
        print("Training with pre-train embedding.")
        x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=False)(inputs_x)
    x = Convolution1D(256, 3, padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    return x

def MCNN(inputs_x, vocab_size, sequence_length, embedding_dim, embedding_matrix=None):
    if embedding_matrix is None:
        print("Training without pre-train embedding.")
        x1 = Embedding(vocab_size, embedding_dim, input_length=sequence_length, trainable=True)(inputs_x)
    else:
        print("Training with pre-train embedding.")
        x1 = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=sequence_length, trainable=False)(inputs_x)
    conv_out = []
    for filter in [2,3,4]:
        conv1 = Convolution1D(256,filter, padding='same')(x1)
        conv1 = Activation('relu')(conv1)
        x_conv1 = GlobalMaxPooling1D()(conv1)
        conv_out.append(x_conv1)

    x = concatenate(conv_out)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    return x

def get_conv_shape(conv):
    return conv.get_shape().as_list()[1:]

class ConvBlockLayer(object):
    """
    two layer ConvNet. Apply batch_norm and relu after each layer
    """

    def __init__(self, input_shape, num_filters):
        self.model = Sequential()
        # first conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # second conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

    def __call__(self, inputs):
        return self.model(inputs)

def VDCNN(inputs, num_quantized_chars, sequence_max_length, embedding_dim, num_filters):

    embedded_sent = Embedding(num_quantized_chars, embedding_dim, input_length=sequence_max_length)(inputs)

    # First conv layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sent)

    # Each ConvBlock with one MaxPooling Layer
    for i in range(len(num_filters)):
        conv = ConvBlockLayer(get_conv_shape(conv), num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

#     # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
#     def _top_k(x):
#         x = tf.transpose(x, [0, 2, 1])
#         k_max = tf.nn.top_k(x, k=top_k)
#         return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))
#     k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

    conv = GlobalMaxPooling1D()(conv)

    # 3 fully-connected layer with dropout regularization
    fc1 = Dropout(0.2)(Dense(2048, activation='relu', kernel_initializer='he_normal')(conv))
    fc2 = Dropout(0.2)(Dense(2048, activation='relu', kernel_initializer='he_normal')(fc1))
    return fc2

def load_data(data_source):
    if data_source == "yelp_review_full_csv" or data_source == "yelp_review_polarity_csv":
        train = pd.read_csv("./data/" + data_source + "/train.csv", header=None, names=["label", "content"], encoding="utf-8")
        test = pd.read_csv("./data/" + data_source + "/test.csv", header=None, names=["label", "content"], encoding="utf-8")
        train.fillna("", inplace=True)
        test.fillna("", inplace=True)
        x_train = train["content"]
        x_test = test["content"]
    
    if data_source == "amazon_review_full_csv" or data_source == "amazon_review_polarity_csv" or data_source == "ag_news_csv" or data_source == "dbpedia_csv" or data_source == "sogou_news_csv":
        train = pd.read_csv("./data/" + data_source + "/train.csv", header=None, names=["label", "title", "content"], encoding="utf-8")
        test = pd.read_csv("./data/" + data_source + "/test.csv", header=None, names=["label", "title", "content"], encoding="utf-8")
        train.fillna("", inplace=True)
        test.fillna("", inplace=True)

        x_train = train["title"] +" " + train["content"]
        x_test = test["title"] + " " + test["content"]
        
    if data_source == "yahoo_answers_csv":
        train = pd.read_csv("./data/" + data_source + "/train.csv", header=None, names=["label", "title", "question", "answer"], encoding="utf-8")
        test = pd.read_csv("./data/" + data_source + "/test.csv", header=None, names=["label", "title", "question", "answer"], encoding="utf-8")
        train.fillna("", inplace=True)
        test.fillna("", inplace=True)

        x_train = train["title"] + " " + train["question"] + " " + train["answer"]
        x_test = test["title"] + " " + test["question"] + " " + test["answer"]

    y_train = train["label"] - 1
    y_test = test["label"] - 1
    y_train = keras.utils.to_categorical(y_train, y_train.max()+1)
    y_test =  keras.utils.to_categorical(y_test, y_test.max()+1)
    
    return x_train.apply(lambda x:x.lower()).values, y_train, x_test.apply(lambda x:x.lower()).values, y_test

def get_variables(data_source, mode):
    pickle_path = "./pickle/" + data_source + "." + mode + ".p"
    if not os.path.exists(pickle_path):
        print("Load data...", data_source)
        x_train, y_train, x_test, y_test = load_data(data_source)
        print("x_train shape:", x_train.shape)
        print("x_test shape:", x_test.shape)

        print ("fit docs to Tokenizer:", time.strftime("%H:%M:%S",time.localtime()))
        if mode == "char":
            t = Tokenizer(char_level=True, filters="\t\n")
        else:
            t = Tokenizer(num_words=vocab_size)
        t.fit_on_texts(np.append(x_train, x_test))
        print ("Done. fit docs to Tokenizer:", time.strftime("%H:%M:%S",time.localtime()))

        print("Convert docs to encoded_docs", time.strftime("%H:%M:%S",time.localtime()))
        train_encoded_docs = t.texts_to_sequences(x_train)
        test_encoded_docs = t.texts_to_sequences(x_test)
        print("Done. Convert docs to encoded_docs", time.strftime("%H:%M:%S",time.localtime()))

        docs = np.append(train_encoded_docs, test_encoded_docs)
        sequence_length = int(np.mean([len(x) for x in docs]) * 2)
        del docs
#         sequence_length = 1014
        print("The max length of sequence is", sequence_length)

        print("Pad encoded_docs", time.strftime("%H:%M:%S",time.localtime()))
        train_padded_docs = pad_sequences(train_encoded_docs, maxlen=sequence_length, padding='post')
        test_padded_docs = pad_sequences(test_encoded_docs, maxlen=sequence_length, padding='post')
        print("Done. Pad encoded_docs", time.strftime("%H:%M:%S",time.localtime()))

        print("Save train_padded_docs, test_padded_docs, tokenizer and sequence_length with pickle")
        cPickle.dump([train_padded_docs, y_train, test_padded_docs, y_test, t, sequence_length], open(pickle_path, "wb"))
        print("Done. Save train_padded_docs, test_padded_docs, tokenizer and sequence_length with pickle")
    else:
        print("Load train_padded_docs, test_padded_docs, tokenizer and sequence_length with pickle")
        train_padded_docs, y_train, test_padded_docs, y_test, t, sequence_length = cPickle.load(open(pickle_path,"rb"))
        print("Done. Load train_padded_docs, test_padded_docs, tokenizer and sequence_length with pickle")
    return train_padded_docs, y_train, test_padded_docs, y_test, t, sequence_length

# if mode == "word":
#     f = open('./data/glove.840B.300d.txt')
#     # load the whole embedding into memory
#     embeddings_index = dict()
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()
#     print("load embedding index")

results = {}
for data_source in data_sources:
    print "run:", data_source
    
    results[data_source] = []
    
    if data_source == "amazon_review_polarity_csv" or data_source == "yelp_review_polarity_csv":
        is_binary_classification = True
    else:
        is_binary_classification = False
    output_activation = "sigmoid" if is_binary_classification else "softmax"
    loss = "binary_crossentropy" if is_binary_classification else "categorical_crossentropy"
    
    train_padded_docs, y_train, test_padded_docs, y_test, t, sequence_length = get_variables(data_source, mode)
    print("The length of sequences is", sequence_length)

#     embedding_matrix = np.zeros((vocab_size, embedding_dim))
#     for word, i in t.word_index.items():
#         if i >= vocab_size: continue
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
    
    for i in range(10):
        inputs_x = Input(shape=(sequence_length,), dtype='int32')
        msdcnn = MsDenselyCNN(inputs_x, vocab_size, sequence_length, embedding_dim)#, embedding_matrix)
    
        outputs = Dense(len(y_train[0]), activation=output_activation)(msdcnn)
        model = Model(inputs=[inputs_x], outputs=outputs)
        model.compile(loss=loss, optimizer='adam', metrics=['acc'])
        
        # print model.summary()
    
        checkpointer = ModelCheckpoint(filepath='./model_'+data_source+'.best.hdf5', save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max')
        model.fit(
            train_padded_docs, y_train, 
            batch_size=batch_size, 
            epochs=num_epochs, 
            validation_data=[test_padded_docs, y_test], 
            verbose=1, 
            callbacks = [EarlyStopping(monitor='val_acc', patience=2, verbose=0),checkpointer]
        )

        model.load_weights("model_"+data_source+".best.hdf5")
        result = model.evaluate(test_padded_docs, y_test, batch_size=batch_size)
        error_rate = 1 - result[1]
        results[data_source].append(error_rate)
        print str(i), ": Error rate of ", data_source, ":", str(error_rate)
        
        del model
        gc.collect()
        K.clear_session()

