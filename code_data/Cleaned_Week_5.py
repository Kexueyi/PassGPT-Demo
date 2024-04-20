#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries



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


# ## Preprocessing



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
print(imdb['review'][3])




words = imdb['review'].apply(lambda x: text_to_word_sequence(x))
stop_words = set(stopwords.words('english'))
filtered_words = words.apply(lambda x: [w for w in x if not w in stop_words])
imdb['review'] = filtered_words.apply(lambda x: " ".join(x))
imdb['review'][3]




imdb.sentiment = imdb.sentiment.apply(lambda x: 1 if x=='positive' else 0)




from sklearn.model_selection import train_test_split
train_review, test_review, train_sent, test_sent = train_test_split(imdb['review'], imdb['sentiment'], test_size=0.25, random_state=42)




#Tfidf vectorizer
tv=TfidfVectorizer(stop_words='english')
#transformed train reviews
train_review_tfidf=np.asarray(tv.fit_transform(train_review).todense())
#transformed test reviews
test_review_tfidf=np.asarray(tv.transform(test_review).todense())
print('Tfidf_train:',train_review_tfidf.shape)
print('Tfidf_test:',test_review_tfidf.shape)




print(train_review_tfidf[3].shape)


# ## SVM Model



from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
#generates a confusion matrix between hand labelled data and model predictions
def getConfMatrix(pred_data, actual):
    conf_mat = confusion_matrix(actual, pred_data, labels=[0,1])
    accuracy = accuracy_score(actual, pred_data)
    precision = precision_score(actual, pred_data, average='micro')
    recall = recall_score(actual, pred_data, average='micro')
    sns.heatmap(conf_mat, annot = True, fmt=".0f", annot_kws={"size": 18})
    print('Accuracy: '+ str(accuracy))
    print('Precision: '+ str(precision))
    print('Recall: '+ str(recall))




from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # RBF kernel

#Train the model using the training sets
clf.fit(train_review_tfidf, train_sent)

#Predict the response for test dataset
y_pred = clf.predict(test_review_tfidf)


# ## Confusion Matrix



getConfMatrix(y_pred,test_sent)




from sklearn.metrics import f1_score
micro = f1_score(test_sent, y_pred, average='micro') 
macro = f1_score(test_sent,y_pred, average='macro')
print('F1 Micro: '+ str(micro))
print('F1 Macro: '+ str(macro))


# ## AUC-ROC

# A logistic regression model is used here as AUC-ROC requires a probabilistic model.



from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
#instantiate the model
log_regression = LogisticRegression()

#fit the model using the training data
log_regression.fit(train_review_tfidf, train_sent)




#define metrics
y_pred_proba = log_regression.predict_proba(test_review_tfidf)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_sent,  y_pred_proba)
auc = metrics.roc_auc_score(test_sent, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# ## BLEU Score



import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
ref = 'The guard arrived late because it was raining.'
cand = 'The guard arrived late because of the rain.'
smoothie = SmoothingFunction().method1
reference = word_tokenize(ref)
candidate = word_tokenize(cand)
weights = (0.25, 0.25, 0.25, 0.25)
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], candidate, weights, smoothing_function=smoothie)
print(BLEUscore)


# ## ROUGE



import evaluate
rouge = evaluate.load('rouge')
predictions = ["Transformers Transformers are fast plus efficient", 
               "Good Morning", "I am waiting for new Transformers"]
references = [
              ["HuggingFace Transformers are fast efficient plus awesome", 
               "Transformers are awesome because they are fast to execute"], 
              ["Good Morning Transformers", "Morning Transformers"], 
              ["People are eagerly waiting for new Transformer models", 
               "People are very excited about new Transformers"]

]
results = rouge.compute(predictions=predictions, references=references)
print(results)


# ## METEOR



from nltk.translate import meteor
from nltk import word_tokenize
score=round(meteor([word_tokenize('The cat sat on the mat')],
                   word_tokenize('The cat was sat on the mat')), 4)
print('The METEOR score is: '+str(score))




from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample sentences for training Word2Vec
sentences = [
    "Word2Vec is a technique for word embedding.",
    "Embedding words in vector space is powerful for NLP.",
    "Gensim provides an easy way to work with Word2Vec.",
]

# Tokenize the sentences into words
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train a Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)  
# Adjust parameters as needed

# Save the trained model for future use
model.save("word2vec.model")

# Load the model (if needed)
# model = Word2Vec.load("word2vec.model")

# Get the word embeddings
word = "word"
if word in model.wv:
    embedding = model.wv[word]
    print(f"Embedding for '{word}': {embedding}")
else:
    print(f"'{word}' is not in the vocabulary.")

# Similarity between words
similarity = model.wv.similarity("word", "embedding")
print(f"Similarity between 'word' and 'embedding': {similarity}")


# ## GloVe



import gensim.downloader as api

# Load the pre-trained GloVe model (you may need to download it first)
glove_model = api.load("glove-wiki-gigaword-100")

# Find the embedding for a specific word
word = "nero"
try:
    embedding = glove_model[word]
    print(f"Embedding for '{word}':")
    print(embedding)
except KeyError:
    print(f"'{word}' is not in the vocabulary.")

# Find the most similar words to a given word
similar_words = glove_model.most_similar(word)
print(f"\nWords most similar to '{word}':")
for similar_word, score in similar_words:
    print(similar_word, score)






