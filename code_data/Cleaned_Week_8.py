#!/usr/bin/env python
# coding: utf-8

# # HyperParams

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
    tf.keras.layers.Dense(1)
])

lstm2 = tf.keras.Sequential([
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




#To implement early Stopping
from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="accuracy", 
                                        mode ="max", patience = 5, 
                                        restore_best_weights = True)





lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(clipvalue=0.5),
              metrics=['accuracy'])

lstm2.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5),
              metrics=['accuracy'])




historyLSTM = lstm.fit(training_set['review'],training_set['sentiment'], epochs=10,
                    validation_data=(test_set['review'],test_set['sentiment']),
                    learning_rate=0.001,
                    batch_size=32,
                    validation_steps=30
                    callbacks=[earlystopping])




historyLSTM2 = lstm2.fit(training_set['review'],training_set['sentiment'], epochs=10,
                    validation_data=(test_set['review'],test_set['sentiment']),
                    batch_size=64,
                    validation_steps=30)




from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score




# Set the number of folds
num_folds = 5
seed = 42

# Initialize StratifiedKFold (or KFold) for cross-validation
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

# Initialize an empty list to store the cross-validation scores
cv_scores = []




X=training_set['review'].values
y=training_set['sentiment'].values
for train_index, test_index in kf.split(X, y):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

    # Compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on the training data
    lstm.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, verbose=0)

    # Evaluate the model on the validation data
    y_pred = (lstm.predict(X_val) > 0.5).astype("int32")
    acc = accuracy_score(y_val, y_pred)

    # Store the cross-validation score
    cv_scores.append(acc)




import statistics
statistics.mean(cv_scores)




import matplotlib.pyplot as plt


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])




plot_graphs(historyLSTM, "accuracy")




plot_graphs(historyLSTM2, "accuracy")






