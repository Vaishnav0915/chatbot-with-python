# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:03:45 2022

@author: vadre
"""

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes=[]
documents=[]
ignore_words=['?','!']
data_file=open('intents.json').read()
intents=json.loads(data_file)
#workng with the text data - tokenizing the data is the primary objective
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizing each word
        w=nltk.word_tokenize(pattern)
        words.extend[w]
        #adding documents in the corpus
        documents.append((w,intent['tag']))
        #add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#lemmatizing: it is grouping and sorting similar kind of words
#objective: lemmatize, lower each word and remove duplicates
words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))
#sorting classes
classes=sorted(list(set(classes)))
#documents:combination b/w patters and intents
print(len(documents),"documents")
#classes = intents
print(len(classes),"classes",classes)
#words=all words, vocabulary 
print(len(words),"unique lemmatized words",words)
pickle.dump(words,open('words.pk1','wb'))
pickle.dump(classes,open('classes.pk1','wb'))
#creating train and test data. training data set will be provided to the input and output.
#computers cannot deal with the text so it is converted into numbers
#creating training data
training=[]
#empty array for output
output_empty=[0]*len(classes)
#training set,bag of words for each sentence
for doc in documents:
    bag=[]
    #list of tokenized words for the pattern
    pattern_words=doc[0]
    #lemmatize each word
    patter_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #create our bag of words with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    #o/p is 0 for each tag and 1 for current tag 
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(bag,output_row)
    
#shuffle our features and turn into np array
random.shuffle(training)
training=np.array(training)
#create train and test lists. X-pattenrs, Y-intents
train_x=list(training[:,0])
train_y=list(training[:,1])
print("training data created")

#building the model
#3 layer model
#layer 1 - 128 neurons
#layer 2 - 64 neurons
#layer 3 - number of neurons
#equal to number of intents to predict output intent with softmax
model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

#compile model
sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov= True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist) 
print("model created")