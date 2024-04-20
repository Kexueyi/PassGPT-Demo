#!/usr/bin/env python
# coding: utf-8

# ## NER



import spacy
from spacy import displacy

NER=spacy.load("en_core_web_sm")




raw_text='''
From 1925 to 1945, Tolkien was the Rawlinson and Bosworth Professor of Anglo-Saxon and a Fellow of Pembroke College, both at the University of Oxford. 
He then moved within the same university to become the Merton Professor of English Language and Literature and Fellow of Merton College, and held these positions from 1945 until his retirement in 1959. 
Tolkien was a close friend of C. S. Lewis, a co-member of the informal literary discussion group The Inklings. 
He was appointed a Commander of the Order of the British Empire by Queen Elizabeth II on 28 March 1972.
'''

text1=NER(raw_text)

for word in text1.ents:
    print(word.text, word.label_)




displacy.render(text1, style='ent',jupyter=True)


# ## POS Tagging



import pandas as pd

text=["You know the greatest lesson of history?",
      "It's that history is whatever the victors say it is.",
      "That's the lesson. Whoever wins, that's who decides the history."]

df= pd.DataFrame(text,columns=['Sentence'])

print(df)




import spacy

#load the small English Model
nlp = spacy.load('en_core_web_sm')

#lists to store tokens and tags
token=[]
pos=[]

for sent in nlp.pipe(df['Sentence']):
    if sent.has_annotation('DEP'):
        #add tokens present in sentnece to token list
        token.append([word.text for word in sent])
        #add POS tags for each token to pos list
        pos.append([word.pos_ for word in sent])




print(df)




print(token)




print(pos)


# ## Dependency Parsing



nlp= spacy.load('en_core_web_sm')

sentence = 'I saw a kitten eating chicken in the kitchen.'

#nlp function returns an obj with individual token information, linguistic features and relations
doc=nlp(sentence)




print('{:<15}|{:<8}|{:<15}|{:<20}'.format('Token','Relation','Head','Children'))
print('-'*70)
for token in doc:
    #Print the token, dependency nature, head and all dependents of the token
    print("{:<15} | {:<18} | {:<15} | {:<20}"
          .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))




#use displacy to render the text
displacy.render(doc, style='dep',jupyter=True, options={'distance':120})






