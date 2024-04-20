#!/usr/bin/env python
# coding: utf-8

# # Preprocessing Techniques

# ### RegEx

# RegEx is especially useful for cleaning out unwanted punctuation marks, captialized letters, special characters etc.



import re


# Eg. Replacing Characters



string='Harper is a good girl.'
#'.' represents any character, while * represents zero or more occurences
#in this case, '..g.*d' will match with 'a good'
#re.sub replaces this with 'the goodest'
re.sub('..g.*d','the goodest',string)


# Using RegEx to remove special chars and punctuation



string='''
One ring to rule them all,
One ring to find them, One ring to bring them all,
and in the darkness, bind them.
'''

string=re.sub('(<.*?>)', ' ', string)
string=re.sub('[,\.!?:()"]', '', string)
string=re.sub('[^a-zA-Z"]',' ',string)

print(string)


# ### Word Tokenization

# Tokenizing a string into individual words



from nltk.tokenize import word_tokenize

words=word_tokenize(string)
print(words)


# ### Stemming



import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce




ps=PorterStemmer()

string='''
From the tip of his wand burst the silver doe. She landed on the office floor, bounded once across the office, and soared out of the window. Dumbledore watched her fly away, and as her silvery glow faded he turned back to Snape, and his eyes were full of tears.
“After all this time?”
“Always,” said Snape.
'''

words=word_tokenize(string)
stemmed_string = reduce(lambda x, y: x +" "+ps.stem(y), words, "")
print(stemmed_string)


# ### Lemmatization

# For the lemmatizer to work as intended, we need to give th lemmatizer the context of each word. This is achieved through POS tagging, which will be covered in greate detail next week. The default POS tagger assumes all words to be nouns if no context is given.



import nltk
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




string = '''
From the tip of his wand burst the silver doe. 
She landed on the office floor, bounded once across the office, and soared out of the window. 
Dumbledore watched her fly away, and as her silvery glow faded he turned back to Snape, and his eyes were full of tears.
“After all this time?”
“Always,” said Snape.”
'''
pos_tagged = nltk.pos_tag(nltk.word_tokenize(string))

wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
print(wordnet_tagged)




lemmatized_sentence = []

for word, tag in wordnet_tagged:
    if tag is None:
        lemmatized_sentence.append(word)
    else:       
        lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
lemmatized_sentence = " ".join(lemmatized_sentence)
 
print(lemmatized_sentence)


# ### Preprocessing for BOW



import nltk
import re
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

text = '''
Never gonna run around and desert you.
'''

dataset= nltk.word_tokenize(text)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W', ' ', dataset[i])
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])
filtered_sentence = [w for w in dataset if not w.lower() in stop_words]
print(filtered_sentence)






