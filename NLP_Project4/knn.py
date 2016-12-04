''' module for knn learner
	http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
	Built from scratch, so may not be optimal. 
	If any error I can use scikit learn and remake this module.
	Author Rommie Choi'''

###################### HOW TO USE THIS CLASS ######################
'''
import knn
import csv
with open("baseline.csv") as csvfile:
	reader = csv.reader(csvfile)
	wordlist = list(reader)

# wordlist would be in a format [(word_0,label_1),...,(word_n,label_n)]

kwv = knn.KnnWordVec(wordlist,k) #k is default 2
kwv.knn_run('likely') = 1 # 1 means cue, -1 not
'''

import math
import operator 
import gensim

# wv = Word2Vec.load_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz", binary=True)

WORD2VEC = '/Users/linoandy/Desktop/Archive/wiki.en.model_20'
# WORD2VEC = '/Users/linoandy/Desktop/GoogleNews-vectors-negative300.bin'

class KnnWordVec:
	def __init__(self, word2labels, k=2):
		self.wvdict= gensim.models.Word2Vec.load(WORD2VEC)
		self.traindata = []
		for word2label in word2labels:
			word = word2label[0]
			label = word2label[1]
			if word in self.wvdict:
				tmparray = self.wvdict[word].tolist()
				tmparray.append(label)
				self.traindata.append(tmparray)
		self.d = len(tmparray)
		self.k = k

		
	def euclideanDistance(self, instance1, instance2, length):
		distance = 0
		for x in range(length):
			distance += pow((instance1[x] - instance2[x]), 2)
		return math.sqrt(distance)

	# Locate most similar neighbours
	# trainingSet = set of training vectors
	# testInstance = vector to be inputted
	# k = how many nearest neighbor?
	def getNeighbors(self, testInstance, k):
		distances = []
		length = len(testInstance)-1
		for x in range(len(self.traindata)):
			dist = self.euclideanDistance(testInstance, self.traindata[x], length)
			distances.append((self.traindata[x], dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(k):
			if x < len(distances):
				neighbors.append(distances[x][0])
		return neighbors

	def getResponse(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

	def knn_run(self, word):
		vec = (self.wvdict[word]).tolist() if word in self.wvdict else [0 for i in range(self.d-1)]
		neighbors = self.getNeighbors(vec, self.k)
		return self.getResponse(neighbors)
