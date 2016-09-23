import re
import glob
import collections
import random
from nltk import tokenize
import nltk
import csv
from operator import itemgetter
from threading import Thread
import math

corpus_word = []
corpus_sentence = []
corpus_sentence_boundary = []

def writetoCSV(newrow, filename):
	filename = filename + ".csv"
	c = csv.writer(open(filename, "a"))
	c.writerow(newrow)
	# c.close()

# read in all the txt file from a specific training data folder, given user's input
# please extract the data.zip 
# and put the extracted data folder in the same folder as this script 
# and then run the code
corpusName = raw_input('Please enter the training data folder name of corpus, for example, autos: ')
path = "./data_corrected/classification task/%s/train_docs/*.txt" % (corpusName)
lambda_1 = float(raw_input('For interpolation, please enter the lambda value of bigram probability, for example, 0.8: '))
lambda_2 = 1 - lambda_1
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

# print sum_unigram

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
# writetoCSV(['---------------------UNIGRAM---------------------'], corpusName)
# for i in range(5):
# 	writetoCSV([unigram_sentence_generator()], corpusName)

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
# writetoCSV(['---------------------BIGRAM----------------------'], corpusName)
# for i in range(5):
# 	writetoCSV([bigram_sentence_generator()], corpusName)
#for i in range(10):
#	print bigram_sentence_generator()

###############################################################################

corpus_boundary_and_unk = []
bigram_with_unk = []
unigram_with_unk = []

# From sentence to list of words
for sentence in corpus_sentence_boundary:
	word_list = sentence.split()
	corpus_boundary_and_unk += word_list

# print corpus_boundary_and_unk

# print len(unigram_with_boundary_counter)
print 'replace with <UNK> ...'

# replace instances of occurrence of 1 with <UNK>
for key, value in unigram_with_boundary_counter.iteritems():
	#print 'BBB'
	if value == 1:
		#print 'BBB'
		for i, val in enumerate(corpus_boundary_and_unk):
			if val == key:
				corpus_boundary_and_unk[i] = '<UNK>'

print 'generate bigram with <UNK> ...'
for i in range(1, len(corpus_boundary_and_unk)):
	# temp for alias; name too long
	temp = corpus_boundary_and_unk
	if (temp[i] == '</s>'):
		continue
	bigram_with_unk.append((temp[i-1], temp[i]))

# print bigram_with_unk

bigram_with_unk_counter = collections.Counter(bigram_with_unk)
# add in the counts fo unseen bigram
# bigram_with_unk_counter['Unseen_Bigram_Never_Exist'] = len(bigram_with_unk) * len(bigram_with_unk) - len(bigram_with_unk)


for word in corpus_boundary_and_unk:
	unigram_with_unk.append(word)

# def generate_unexisted_bigram (word1, unigram_list):
# 	for word2 in unigram_list:
# 		# if ending with <s>, skip
# 		if word2 == '<s>':
# 			continue
# 		if bigram_with_unk_counter.get((word1, word2)) == None:
# 			bigram_with_unk_counter[(word1, word2)] = 0
# 			# print (word1, word2)


# list_threads = []
# for word in unigram_with_unk:
# 	# if starting with </s>, skip
# 	if word == '</s>':
# 		continue

# 	t = Thread(target=generate_unexisted_bigram, args=(word, unigram_with_unk))
# 	list_threads.append(t)
# 	t.start()
# 	if len(list_threads) > 50:
# 		for j in list_threads:
# 			j.join()
# 		list_threads = []

	# for word2 in unigram_with_unk:
	# 	# if ending with <s>, skip
	# 	if word2 == '<s>':
	# 		continue
	# 	if bigram_with_unk_counter.get((word, word2)) == None:
	# 		bigram_with_unk_counter[(word, word2)] = 0
	# 		print (word, word2)

unigram_with_unk_counter = collections.Counter(unigram_with_unk)
# print unigram_with_unk
# print bigram_with_unk_counter








###############################################################################

# print 'unigram - number of token: ', len(corpus_word)
# print 'unigram - number of word type: ', len(unigram_counter)

# print 'bigram - number of token: ', len(bigram_list)
# print 'bigram - number of word type: ', len(bigram_counter)

# Unknown words: replacing all words that occur only once with "<UNK>"
def unknown_word_processor (words_counter):
	unknown_word_counter = {}
	for key, value in words_counter.iteritems():
		unknown_word_counter.setdefault('<UNK>', 0)
		if value == 1:
			unknown_word_counter['<UNK>'] += 1
		else:
			unknown_word_counter[key] = value
	return unknown_word_counter

# print unknown_word_processor(bigram_counter)

# generate a dictionary containing count of word counts, for Good-Turing
def count_of_counts (original_word_counter, original_n_gram, n_gram_type):
	counts_counter_temp = {}
	for key, value in original_word_counter.iteritems():
		counts_counter_temp.setdefault(str(value), 0)
		counts_counter_temp[str(value)] += 1
	
	# add in the counts fo unseen bigram
	if n_gram_type == 'bigram':
		counts_counter_temp['0'] = len(original_n_gram) * len(original_n_gram) - len(original_n_gram)

	counts_counter = []
	for key, value in counts_counter_temp.iteritems():
		counts_counter.append((int(key), value))

	counts_counter = sorted(counts_counter, key=itemgetter(0))
	return counts_counter


# print count_of_counts(bigram_with_unk_counter, bigram_with_unk, 'bigram')

# Good-Turing
def good_turing (counter_of_counts, original_word_counter, n_gram_type):
	new_counter_of_counts = []
	for i in range(len(counter_of_counts)):
		if counter_of_counts[i][0] < 5:
			new_count = (counter_of_counts[i][0] + 1) * counter_of_counts[i][1] / float(counter_of_counts[i+1][1])
			new_counter_of_counts.append((new_count, counter_of_counts[i][0], counter_of_counts[i][1]))
		else:
			new_count = counter_of_counts[i][0]
			new_counter_of_counts.append((new_count, counter_of_counts[i][0], counter_of_counts[i][1]))

	good_turing_counter = {}
	for new_counter in new_counter_of_counts:
		# add in one key-value pair to stand for adjusted counts of unseen bigram in corpus
		if new_counter[1] == 0 and n_gram_type == 'bigram':
			good_turing_counter['Unseen_Bigram_Never_Exist'] = new_counter[0]
		# adjust counts of existing bigram
		for key, value in original_word_counter.iteritems():
			if int(value) == new_counter[1]:
				good_turing_counter[key] = new_counter[0]

	# return new_counter_of_counts
	return good_turing_counter

bigram_counter_good_turing = good_turing(count_of_counts(bigram_with_unk_counter, bigram_with_unk, 'bigram'), bigram_with_unk_counter, 'bigram')
unigram_counter_good_turing = good_turing(count_of_counts(unigram_with_unk_counter, unigram_with_unk, 'unigram'), unigram_with_unk_counter, 'unigram')

sum_unigram_token_good_turing = 0
for key, value in unigram_counter_good_turing.iteritems():
	sum_unigram_token_good_turing += value

# for key, value in good_turing(count_of_counts(bigram_with_unk_counter, bigram_with_unk, 'bigram'), bigram_with_unk_counter, 'bigram').iteritems():
# 	print key, value

######################### Perplexity ############################

# process the test data files
csv_data_dump_bigram = [corpusName+'_bigram']
csv_data_dump_unigram = [corpusName+'_unigram']
csv_data_dump_interpolation = [corpusName+'_interpolation']
csv_data_dump_filename = ['']
path_test_data = "./data_corrected/classification task/test_for_classification/*.txt"
print "length of", len(glob.glob(path_test_data))
for filename_test_data in glob.glob(path_test_data):
	with open(filename_test_data, 'r') as g:
		for line in g:
			# find "From : email" and replace them with empty string
			email = re.findall(r"From\s*:\s*[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s", line)
			if len(email) > 0:
				line = line.replace(email[0].encode('utf-8'), '', 1)

			# find first "Subject : " and replace with empty string
			line = line.replace('Subject : ', '', 1)

			# find ' >' and replace with empty string
			line = line.replace(' >', '')

			# build unigram from test data corpus
			unigram_test_data = line.split(' ')

			# build corpus of sentences
			corpus_sentence_test_data = tokenize.sent_tokenize(line)
			for index, item in enumerate(corpus_sentence_test_data):
				corpus_sentence_test_data[index] = '<s> ' + item + ' </s>'

			# build bigram from test data corpus
			corpus_test_data = ''
			for sentence_test_data in corpus_sentence_test_data:
				corpus_test_data += (' ' + sentence_test_data)
			corpus_test_data_word_list = corpus_test_data.split(' ')


			bigram_test_data = []
			for i in range(1, len(corpus_test_data_word_list)):
				# temp for alias; name too long
				temp = corpus_test_data_word_list
				if (temp[i] == '</s>'):
					continue
				bigram_test_data.append((temp[i-1], temp[i]))

			# print unigram_test_data, bigram_test_data

			sum_negative_log_bigram = 0
			for bigram in bigram_test_data:
				if bigram_counter_good_turing.get(bigram) == None:
					count_bigram_good_turing = float(bigram_counter_good_turing['Unseen_Bigram_Never_Exist'])
				else:
					count_bigram_good_turing = float(bigram_counter_good_turing[bigram])

				if unigram_counter_good_turing.get(bigram[0]) == None:
					count_unigram_good_turing = unigram_counter_good_turing['<UNK>']
				else:
					count_unigram_good_turing = unigram_counter_good_turing[bigram[0]]
				
				sum_negative_log_bigram += ( - math.log(count_bigram_good_turing / count_unigram_good_turing))
			# print 'sum_negative_log_bigram', sum_negative_log_bigram
			perplexity_bigram = math.exp(sum_negative_log_bigram / len(bigram_test_data))
			# print 'perplexity_bigram', perplexity_bigram
			csv_data_dump_bigram.append(perplexity_bigram)
			print filename_test_data

			sum_negative_log_unigram = 0
			for unigram in unigram_test_data:
				if unigram_counter_good_turing.get(unigram) == None:
					count_unigram_good_turing = float(unigram_counter_good_turing['<UNK>'])
				else:
					count_unigram_good_turing = float(unigram_counter_good_turing[unigram])

				sum_negative_log_unigram += ( - math.log(count_unigram_good_turing / sum_unigram_token_good_turing))

			perplexity_unigram = math.exp(sum_negative_log_unigram / len(unigram_test_data))
			csv_data_dump_unigram.append(perplexity_unigram)

			# interpolation
			sum_negative_log_interpolation = 0
			for bigram in bigram_test_data:
				if bigram_counter_good_turing.get(bigram) == None:
					count_bigram_good_turing = float(bigram_counter_good_turing['Unseen_Bigram_Never_Exist'])
				else:
					count_bigram_good_turing = float(bigram_counter_good_turing[bigram])

				if unigram_counter_good_turing.get(bigram[0]) == None:
					count_unigram_good_turing = unigram_counter_good_turing['<UNK>']
				else:
					count_unigram_good_turing = unigram_counter_good_turing[bigram[0]]

				interpolated_probability = lambda_1 * (count_bigram_good_turing / count_unigram_good_turing) + lambda_2 * (count_unigram_good_turing / sum_unigram_token_good_turing)

				sum_negative_log_interpolation += ( - math.log(interpolated_probability))

			perplexity_interpolation = math.exp(sum_negative_log_interpolation / len(bigram_test_data))
			csv_data_dump_interpolation.append(perplexity_interpolation)
			csv_data_dump_filename.append(re.findall('file_[0-9]+', filename_test_data)[0].encode('utf-8'))
			

# transpose and print to csv
l = [csv_data_dump_filename, csv_data_dump_unigram, csv_data_dump_bigram, csv_data_dump_interpolation]
l = zip(*l)
for item in l:
	writetoCSV(item, './smoothing_perplexity/'+corpusName+str(lambda_1))





