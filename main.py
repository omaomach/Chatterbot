import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() #will be used to stem our words in order to get the root of the word. This will help improve on the accuracy of the model

import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import pickle

with open("intent.json") as file:
	data = json.load(file) #we are loading th intents file

try:
	x
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = [] #we are creating our words list
	labels = []
	docs_x = [] # the reason for these two is that for each word, we want to know its tag
	docs_y = []

	for intent in data ["intent"]: #for looping through all data in the intents file
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern) #to get all the single words in our patters in order to stem them.
			words.extend(wrds) #we are putting all our tokenized words in our words list
			docs_x.append(wrds)
			docs_y.append(intent["tag"]) #

		if intent["tag"] not in labels:
			labels.append(intent["tag"]) # we are getting the individual tags in the intents file.

	words = [stemmer.stem(w.lower()) for w in words if w != "?"] #we stemming the words in the words list and removing duplicates
	words = sorted(list(set(words))) #duplicates removed and words sorted

	labels = sorted(labels) # sorting the labels

	training = [] # has a bunch of bags of words
	output = [] # a list of 0s and 1s

	out_empty = [0 for _ in range(len(labels))] 

	for x, doc in enumerate(docs_x):
		bag = [] # creating a bag of words

		wrds = [stemmer.stem(w) for w in doc] # stemming our words in the wrds list

		for  w in words :
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0) # we are through all our words and putting 1s and 0s

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1 # looping through the labels list to find out where the required tag is at.

		training.append(bag)
		output.append(output_row)

	training = np.array(training)
	output = np.array(output) # we are changing our output and training data into numpy array which is the form that is taken by our model

	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)


tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) #define the input shape we are expecting for our model
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # we are getting the probabilities for each output
net = tflearn.regression(net) # this model is simply predicting which tag has the users response

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) # we are passing the model all of our training data
model.save("model.tflearn") # saving the model

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s) # we are creating a list of tokenized words in s_words
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se: # the current word we are looking at in the words list is equal to the word in your sentence
				bag[i] = 1 # the word exits

	return np.array(bag) # converting the bag of words it a numpy array
				

def chat():
	print("Start Talking with Bot(Type quit to stop)")
	while  True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		results = model.predict([bag_of_words(inp, words)])[0]
		results_index = np.argmax(results)
		tag = labels[results_index]

		if results [results_index] > 0.7:
			for tg in data["intent"]:
				if tg['tag'] == tag:
					responses = tg['responses']

			
			print(random.choice(responses))

		else:
			print("I didn't get that :( Try again!")
			

chat()		

			



		
		
