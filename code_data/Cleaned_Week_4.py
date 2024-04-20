#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import re
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


# ## Reading the dataset



imdb_df = pd.read_csv(r"C:\Users\liuru\Desktop\EE6405\Data\IMDB\IMDB Dataset.csv")




df_positive = imdb_df[imdb_df['sentiment']=='positive'][:5000]
df_negative = imdb_df[imdb_df['sentiment']=='negative'][:5000]
imdb = pd.concat([df_positive,df_negative ])




imdb.shape


# ## Preprocessing

# Data before preprocessing



print(imdb['review'][3])


# Preprocessing with regex to remove punctuation and special chars, lowercase



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


# POS Tagging + Lemmatizer



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




print(imdb['review'][3])


# Remove Stop Words



words = imdb['review'].apply(lambda x: text_to_word_sequence(x))
stop_words = set(stopwords.words('english'))
filtered_words = words.apply(lambda x: [w for w in x if not w in stop_words])
imdb['review'] = filtered_words.apply(lambda x: " ".join(x))




imdb['review'][3]




imdb.sentiment = imdb.sentiment.apply(lambda x: 1 if x=='positive' else 0)




from sklearn.model_selection import train_test_split
train_review, test_review, train_sent, test_sent = train_test_split(imdb['review'], imdb['sentiment'], test_size=0.25, random_state=42)




print(train_review.head)




#Tfidf vectorizer
tv=TfidfVectorizer(stop_words='english')
#transformed train reviews
train_review_tfidf=np.asarray(tv.fit_transform(train_review).todense())
#transformed test reviews
test_review_tfidf=np.asarray(tv.transform(test_review).todense())
print('Tfidf_train:',train_review_tfidf.shape)
print('Tfidf_test:',test_review_tfidf.shape)




print(train_review_tfidf[3].shape)


# ## Classifiers



from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
#generates a confusion matrix between hand labelled data and model predictions
def getConfMatrix(pred_data, actual):
    conf_mat = confusion_matrix(actual, pred_data, labels=[0,1]) 
    micro = f1_score(actual, pred_data, average='micro') 
    macro = f1_score(actual,pred_data, average='macro')
    sns.heatmap(conf_mat, annot = True, fmt=".0f", annot_kws={"size": 18})
    print('F1 Micro: '+ str(micro))
    print('F1 Macro: '+ str(macro))


# Support Vector Machines



from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # RBF kernel

#Train the model using the training sets
clf.fit(train_review_tfidf, train_sent)

#Predict the response for test dataset
y_pred = clf.predict(test_review_tfidf)




getConfMatrix(y_pred,test_sent)


# Extreme Learning Machines



from skelm import ELMClassifier
clf = ELMClassifier()
#Train the model using the training sets
clf.fit(train_review_tfidf, train_sent)

#Predict the response for test dataset
y_pred = clf.predict(test_review_tfidf)




getConfMatrix(y_pred,test_sent)


# Gaussian Process



from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
kernel = 1.0 * RBF(1.0)
clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
#Train the model using the training sets
clf.fit(train_review_tfidf, train_sent)

#Predict the response for test dataset
y_pred = clf.predict(test_review_tfidf)




getConfMatrix(y_pred,test_sent)


# KMeans Clustering



from sklearn.cluster import KMeans
for seed in range(5):
    kmeans = KMeans(
        n_clusters=2,
        max_iter=100,
        n_init=1,
        random_state=seed,
    ).fit(train_review_tfidf)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    print(f"Number of elements assigned to each cluster: {cluster_sizes}")
print()




original_space_centroids = kmeans.cluster_centers_
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = tv.get_feature_names_out()

for i in range(2):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]} ", end="")
    print()




from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
train_review_lsa = lsa.fit_transform(train_review_tfidf)
explained_variance = lsa[0].explained_variance_ratio_.sum()
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")




kmeans = KMeans(
    n_clusters=2,
    max_iter=100,
    n_init=5,
    random_state=seed,
).fit(train_review_lsa)




original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = tv.get_feature_names_out()

for i in range(2):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]} ", end="")
    print()






