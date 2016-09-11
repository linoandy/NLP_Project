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

# randomly select a unigram from corpus
def rand_select_unigram ():
	# random integer [0..(counter_sum - 1)]
	rand_int = random.randrange(0, unigram_counter_sum)
	temp = 0
	for key in unigram_counter:
		temp += unigram_counter[key]
		if rand_int < temp:
			return key

# generate one sentence from unigrams
def unigram_sentence_generator ():
	sentence = ''
	while True:
		rand_word = rand_select_unigram ()
		sentence += rand_word + " "
		if rand_word == "." or rand_word == "!" or rand_word == "?":
			return sentence

print '---------------------UNIGRAM---------------------'
writetoCSV(['---------------------UNIGRAM---------------------'], corpusName)
for i in range(5):
	writetoCSV([unigram_sentence_generator()], corpusName)

# if probabilities are needed:
#for key in counter:
#	counter[key] /= float(counter_sum)

### BIGRAM RANDOM SENTENCE GENERATION ###

bigram_list = []

# build bigram list from boundary-applied sentences
for sentence in corpus_sentence_boundary:
	word_list = sentence.split(' ')
	bigram_in_sentence = []
	for i in range(1, len(word_list)):
		# append tuple (prev, next)
		bigram_in_sentence.append((word_list[i-1], word_list[i]))
	# add bigrams from each sentence to the bigram list
	bigram_list += bigram_in_sentence

# randomly select a bigram, given the previous word
def rand_select_bigram (given_word):
	# selecting only the bigrams that start with the given word
	selected_bigram_list = [x for x in bigram_list if x[0] == given_word]
	counter = collections.Counter(selected_bigram_list)
	counter_sum = sum(counter.values())
	# random integer [0..(counter_sum - 1)]
	rand_int = random.randrange(0, counter_sum)
	temp = 0
	for key in counter:
		temp += counter[key]
		if rand_int < temp:
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

