#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data1 = "I'm designing a document and don't want to get bogged down in what the text actually says"
data2 = "I'm creating a template for various paragraph styles and need to see what they will look like."
data3 = "I'm trying to learn more about some features of Microsoft Word and don't want to practice on a real document"

df1 = pd.DataFrame({'First_Para':[data1], 'Second_Para':[data2], 'Third_Para':[data3]})
vectorizer = TfidfVectorizer()
doc_vec = vectorizer.fit_transform(df1.iloc[0])

df2 = pd.DataFrame(doc_vec.toarray().transpose(),index=vectorizer.get_feature_names_out())

df2.columns = df1.columns
print(df2)


# ## BM25



from rank_bm25 import BM25Okapi

corpus=[
    "I will take the ring, though I do not know the way.",
    "I will help you bear this burden, Frodo Baggins, as long as it is yours to bear",
    "If by my life or death I can protect you, I will."
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)




query="I will take"
tokenized_query = query.split(" ")
doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)


# ## LDA



# Importing modules
import pandas as pd
# Read Corona Tweets
tweets = pd.read_csv(r"C:\Users\liuru\Desktop\EE6405\Data\archive\Corona_NLP_train.csv", encoding='latin-1')
# Print head
tweets.head()




# Remove the columns
tweets = tweets.drop(columns=['ï»¿UserName', 'ScreenName', 'Location','TweetAt','Sentiment'], axis=1).sample(100)
# Print out the first rows of papers
tweets.head()




# Load the regular expression library
import re
# Remove punctuation
tweets['OriginalTweet_processed'] = \
tweets['OriginalTweet'].map(lambda x: re.sub('[@#,\.!?]', '', x))
# Convert the tweets to lowercase
tweets['OriginalTweet_processed'] = \
tweets['OriginalTweet_processed'].map(lambda x: x.lower())
# Print out the first rows of tweets
tweets['OriginalTweet_processed'].head()




import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['https', 'tco'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = tweets.OriginalTweet_processed.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])




import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])




from pprint import pprint
# number of topics
num_topics = 7
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]




import pyLDAvis.gensim
import pickle 
import pyLDAvis
import os
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

LDAvis_prepared


# ## LSA



import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt




import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['https', 'tco'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
def stem_words(texts):
    return [[p_stemmer.stem(word) for word in simple_preprocess(str(doc))
             ] for doc in texts]

data = tweets.OriginalTweet_processed.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])




doc_term_matrix = [id2word.doc2bow(twt) for twt in data_words]




lsa_model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = id2word)
print(lsa_model.print_topics(num_topics=num_topics, num_words=10))


# ## NMF



import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
 
tweets = pd.read_csv(r"C:\Users\liuru\Desktop\EE6405\Data\archive\Corona_NLP_train.csv", encoding='latin-1')
# Print head
tweets.head()




# Remove the columns
tweets = tweets.drop(columns=['ï»¿UserName', 'ScreenName', 'Location','TweetAt','Sentiment'], axis=1).sample(100)
# Print out the first rows of papers
tweets.head()




# use tfidf by removing tokens that don't appear in at least 50 documents
vect = TfidfVectorizer(min_df=10, stop_words=stop_words )
 
# Fit and transform
X = vect.fit_transform(tweets.OriginalTweet)




# Create an NMF instance: model
# the 10 components will be the topics
model = NMF(n_components=10, random_state=5)
 
# Fit the model to TF-IDF
model.fit(X)
 
# Transform the TF-IDF: nmf_features
nmf_features = model.transform(X)




components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names_out())
components_df




for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    print(f'For topic {topic+1} the words with the highest value are:')
    print(tmp.nlargest(10))
    print('\n')


# ## PCA



from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['https','tco'])




vect= TfidfVectorizer(stop_words=stop_words)
x = vect.fit_transform(tweets.OriginalTweet)
tf_idf_vect = pd.DataFrame(x.toarray().transpose(),index=vect.get_feature_names_out())




tf_idf_vect




from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit_transform(tf_idf_vect)




print(pca.components_)


# ## SVD



import numpy as np




vect= TfidfVectorizer(stop_words=stop_words, smooth_idf=True)
x = vect.fit_transform(tweets.OriginalTweet).todense()
x = np.asarray(x)




from sklearn.decomposition import TruncatedSVD
svd_modeling = TruncatedSVD(n_components=4, algorithm='randomized', n_iter=100, random_state=122)
svd_modeling.fit(x)
components=svd_modeling.components_
vocab = vect.get_feature_names_out()




topic_word_list = []
def get_topics(components):
    for i, comp in enumerate(components):
        terms_comp =zip(vocab,comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        topic=" "
        for t in sorted_terms:
            topic= topic + ' ' + t[0]
        topic_word_list.append(topic)
        print(topic_word_list)
    return topic_word_list
get_topics(components)
        






