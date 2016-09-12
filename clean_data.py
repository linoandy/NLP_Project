import re
import glob
import collections
import random
from nltk import tokenize
import nltk
import csv

corpus_word = []
corpus_sentence = []
corpus_sentence_boundary = []

def writetoCSV(newrow, filename):
	filename = filename + ".csv"
	c = csv.writer(open(filename, "a"))
	c.writerow(newrow)
	# c.close()

corpusName = raw_input('Please enter the type of corpus: ')
path = "./data_corrected/classification_task/%s/train_docs/*.txt" % (corpusName)
for filename in glob.glob(path):
	with open(filename, 'r') as f:
		for line in f:
			# find "From : email" and replace them with empty string
			email = re.findall(r"From\s*:\s*[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s", line)
			if len(email) > 0:
				line = line.replace(email[0].encode('utf-8'), '', 1)

			# find first "Subject : " and replace with empty string
			line = line.replace('Subject : ', '', 1)

			# find ' >' and replace with empty string
			line = line.replace(' >', '')

			# build corpus of sentences
			corpus_sentence += tokenize.sent_tokenize(line)

			# build corpus of words
			corpus_word += line.split(' ')

#print corpus_word # this is a list of words
#print corpus_sentence # this is a list of sentences

# create a corpus_sentence with <s> and </s>
for sentence in corpus_sentence:
	sentence = '<s> ' + sentence.encode('utf-8') + ' </s>'
	corpus_sentence_boundary.append(sentence)

#print corpus_sentence_boundary # this is a list of sentences with <s> and </s>

##############################################################

### UNIGRAM RANDOM SENTENCE GENERATION ###

unigram_counter = collections.Counter(corpus_word)
unigram_counter_sum = sum(unigram_counter.values())
unigram_probabilities = dict()

# for each unigram, compute the probability of each and store in dictionary
for key in unigram_counter:
	unigram_probabilities[key] = unigram_counter[key] / float(unigram_counter_sum)

sum_unigram = 0
for key in unigram_probabilities:
	sum_unigram += unigram_probabilities[key]

print sum_unigram

# randomly select a unigram from corpus
def rand_select_unigram():
	# [0..1)
	rand_float = random.random()
	temp = 0
	for key in unigram_probabilities:
		temp += unigram_probabilities[key]
		if rand_float < temp:
			return key

# generate one sentence from unigrams
def unigram_sentence_generator ():
	sentence = ''
	while True:
		rand_word = rand_select_unigram ()
		sentence += rand_word + ' '
		if rand_word == "." or rand_word == "!" or rand_word == "?":
			return sentence

print '---------------------UNIGRAM---------------------'
writetoCSV(['---------------------UNIGRAM---------------------'], corpusName)
for i in range(5):
	writetoCSV([unigram_sentence_generator()], corpusName)

#for i in range(10):
#	print unigram_sentence_generator()

### BIGRAM RANDOM SENTENCE GENERATION ###

bigram_list = []
# unigram list with <s> and </s>
unigram_list_with_boundary = []

# build bigram list from boundary-applied sentences
for sentence in corpus_sentence_boundary:
	word_list = sentence.split(' ')
	bigram_in_sentence = []
	unigram_in_sentence = []
	for i in range(1, len(word_list)):
		# append tuple (prev, next)
		bigram_in_sentence.append((word_list[i-1], word_list[i]))
	for i in range(0, len(word_list)):
		unigram_in_sentence.append(word_list[i])
	# add bigrams, unigrams from each sentence to the cumulative lists
	bigram_list += bigram_in_sentence
	unigram_list_with_boundary += unigram_in_sentence

bigram_counter = collections.Counter(bigram_list)
unigram_with_boundary_counter = collections.Counter(unigram_list_with_boundary)

# a dictionary of bigram probabilities
# e.g. {('I', 'am') : 0.3} means that P(am | I) = 0.3.
bigram_probabilities = dict()
for key in bigram_counter:
	bigram_probabilities[key] = bigram_counter[key] / float(unigram_with_boundary_counter[key[0]])

# randomly select a bigram, given the previous word
def rand_select_bigram (given_word):
	# selecting only the bigram probabilities that start with the given word
	selected_bigram_probabilities = \
	dict((k,v) for k,v in bigram_probabilities.items() if k[0] == given_word)
	rand_float = random.random()
	temp = 0
	for key in selected_bigram_probabilities:
		temp += selected_bigram_probabilities[key]
		if rand_float < temp:
			# return only the newly selected word of the tuple
			return key[1]

# generates one random bigram sentence
def bigram_sentence_generator ():
	sentence = '<s> '
	given_word = '<s>'
	while True:
		rand_word = rand_select_bigram(given_word)
		sentence += rand_word + ' '
		if(rand_word == '</s>'):
			return sentence
		given_word = rand_word

print '---------------------BIGRAM----------------------'
writetoCSV(['---------------------BIGRAM----------------------'], corpusName)
for i in range(5):
	writetoCSV([bigram_sentence_generator()], corpusName)
#for i in range(10):
#	print bigram_sentence_generator()
