#!/usr/bin/env python
# coding: utf-8

# # Week 6

# ## Importing Libraries



import pandas as pd
import re
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


# ## Data Preprocessing

# We will continue to use the IMDB dataset for this week's examples.



imdb_df = pd.read_csv(r"C:\Users\liuru\Desktop\EE6405\Data\IMDB\IMDB Dataset.csv")
df_positive = imdb_df[imdb_df['sentiment']=='positive'][:5000]
df_negative = imdb_df[imdb_df['sentiment']=='negative'][:5000]
imdb = pd.concat([df_positive,df_negative ])
imdb.shape




# remove "(<.*?>)" markup
imdb['review'] = imdb['review'].apply(lambda x: re.sub('(<.*?>)', ' ', x))
# remove punctuation marks 
imdb['review'] = imdb['review'].apply(lambda x: re.sub('[,\.!?:()"]', '', x))
# remove whitespace
imdb['review'] = imdb['review'].apply(lambda x: x.strip())
# remove all strings that contain a non-letter
imdb['review'] = imdb['review'].apply(lambda x: re.sub('[^a-zA-Z"]',' ',x))
# convert to lower
imdb['review'] = imdb['review'].apply(lambda x: x.lower())




nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
lemmatizer= WordNetLemmatizer()
from nltk.corpus import wordnet




def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
def tagged_lemma(string):
    
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(string))

    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmatized_sentence = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:       
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    return lemmatized_sentence




imdb['review']=imdb['review'].apply(tagged_lemma)




imdb.sentiment = imdb.sentiment.apply(lambda x: 1 if x=='positive' else 0)




from sklearn.model_selection import train_test_split
training_set, test_set= train_test_split(imdb, test_size=0.25, random_state=42)




training_set


# ## RNNs

# Importing libraries:
# We will use the tensorflow library to conctruct out neural models.



import numpy as np
import tensorflow as tf


# ### Text Encoder:
# The raw text loaded by tfds needs to be processed before it can be used in a model. The simplest way to process text for training is using the TextVectorization layer. 



reviews = training_set['review'].tolist()




VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    pad_to_max_tokens=True
)
encoder.adapt(reviews)

vocab = np.array(encoder.get_vocabulary())




vocab[:20]




training_set.review.iloc[0]




encoded_example = encoder(training_set.review.iloc[3]).numpy()
encoded_example


# ### Unidirectional RNN:
# To create a unidirectional RNN, we build the model as a tf.keras.Sequential.
# 
# The first layer is the encoder, which converts the text to a sequence of token indices.
# 
# After the encoder is an embedding layer. An embedding layer stores one vector per word. When called, it converts the sequences of word indices to sequences of vectors. These vectors are trainable. After training (on enough data), words with similar meanings often have similar vectors.
# 
# This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a tf.keras.layers.Dense layer.
# 
# A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input on the next timestep.
# 
# After the RNN has converted the sequence to a single vector the two layers.Dense do some final processing, and convert from this vector representation to a single logit as the classification output.



simpleRNN = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# We use the Binary Crossentropy Loss since we are dealing with Binary Classification. We also use the Adam optmizer. We use a smaller learning rate.



simpleRNN.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])




historyRNN = simpleRNN.fit(training_set['review'],training_set['sentiment'], epochs=10,
                    validation_data=(test_set['review'],test_set['sentiment']), 
                    validation_steps=30)


# ### LSTM:
# To modify the code and create an LSTM model, we simply change the RNN layer to an LSTM layer.



lstm = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.LSTM(64), #Replacing the simple RNN with a LSTM
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])




lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])




historyLSTM = lstm.fit(training_set['review'],training_set['sentiment'], epochs=10,
                    validation_data=(test_set['review'],test_set['sentiment']), 
                    validation_steps=30)


# ### Bi-Directional LSTM:
# To implement the Bi-directional LSTM, we modify the code as follows:



biDir = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), #Replacing the LSTM with a bidirectional LSTM
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])




biDir.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])




historybiDir = biDir.fit(training_set['review'],training_set['sentiment'], epochs=10,
                    validation_data=(test_set['review'],test_set['sentiment']), 
                    validation_steps=30)


# ### GRUs:
# Similarly, to create a GRU, we simply replace the corresponding layer.



gru = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.GRU(64),  # Replace with a GRU layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])




gru.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])




historyGRU = gru.fit(training_set['review'],training_set['sentiment'], epochs=10,
                    validation_data=(test_set['review'],test_set['sentiment']), 
                    validation_steps=30)


# The matplotlib library will be used to track our model's metrics



import matplotlib.pyplot as plt


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])




plot_graphs(historyRNN, "accuracy")




plot_graphs(historyLSTM, "accuracy")




plot_graphs(historybiDir, "accuracy")




plot_graphs(historyGRU, "accuracy")






