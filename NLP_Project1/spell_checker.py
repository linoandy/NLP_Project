import re
import glob
import nltk
import collections
import codecs
import os
from operator import itemgetter


#################################################################################
# TO RUN, put the spell checking data as "spell_checking_task_v2" under ./data_corrected




corpus_names = ['atheism','autos','graphics','medicine','motorcycles','religion','space']
#corpus_names = ['temp']
#corpus_names = ['atheism']


confusion_set = []
# getting rid of UTF-8 BOM at the beginning, converting to ascii
f = codecs.open('./data_corrected/spell_checking_task_v2/confusion_set.txt', encoding='ascii', errors='ignore')
for line in f:
	words = line.split()
	for i in range(1):
		# remove end-of-line markers
		words[i].replace('\r\n', '')
	# insert confusion set as tuples
	if words[0] == 'maybe':
		# special treatment for may be
		words[1] = 'mayxyzbe'
	confusion_set.append((words[0], words[1]))
#print confusion_set

def preprocess(corpus_name):
	corpus_sentence = []
	corpus_word = []
	path_train = "./data_corrected/spell_checking_task_v2/%s/train_docs/*.txt" % corpus_name
	glob_temp = glob.glob(path_train)
	glob_length = len(glob_temp)
	total_doc_number = glob_length
	train_length = int(round(glob_length * 0.8))
	development_length = glob_length - train_length
	for i in range(train_length):
		with open(glob.glob(path_train)[i], 'r') as f:
			for line in f:
				# find "From : email" and replace them with empty string
				email = re.findall(r"From\s*:\s*[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s", line)
				if len(email) > 0:
					line = line.replace(email[0].encode('utf-8'), '', 1)

				# find first "Subject : " and replace with empty string
				line = line.replace('Subject : ', '', 1)

				# special treatment for may be
				line = line.replace('may be', 'mayxyzbe')
				line = line.replace('May be', 'Mayxyzbe')
				

				# find ' >' and replace with empty string
				line = line.replace(' >', '')

				# build corpus of sentences
				corpus_sentence += nltk.tokenize.sent_tokenize(line)

				# build corpus of words
				corpus_word += line.split(' ')
	#print corpus_word
	return corpus_sentence, corpus_word, total_doc_number, train_length

def generate_sentence_markers(corpus_sentence):
	result = []
	for sentence in corpus_sentence:
		sentence = 	'<s> ' + sentence.encode('utf-8') + ' </s>'
		result.append(sentence)

	return result

########################################################################
########################################################################
########################################################################
########################################################################

def generate_unigrams(corpus_word):
	unigram_counter = collections.Counter(corpus_word)
	unigram_sum = sum(unigram_counter.values())
	unigram_probabilities = {}
	for key in unigram_counter:
		unigram_probabilities[key] = unigram_counter[key] / float(unigram_sum)
	return corpus_word, unigram_counter, unigram_probabilities

# returns 0 if there are no matching keys
def get_value(key, dict):
	if(dict.get(key) == None):
		return 0
	else:
		return dict.get(key)

# a word is either all lower case, contains uppercase, or non-alphabetical
# 0 for non-alphabetical, 1 for all lower case, 2 for containing uppercase
def word_classification(word):
	is_alpha = word.isalpha()
	word_lower = word.lower()
	if is_alpha == False:
		return 0
	elif word_lower == word:
		return 1
	else:
		return 2

# returns the unigram probability for each word
def unigram_prob_word(word, word_class, prob):
	if(word_class == 0):
		return get_value(word, prob)
	# if lowercase word
	elif(word_class == 1):
		# makes first letter upper
		word_upper = word.title()
		return get_value(word, prob) + get_value(word_upper, prob)
	# uppercase	word
	else:
		word_lower = word.lower()
		return get_value(word, prob) + get_value(word_lower, prob)

# invariant : confusion word lowercase
def get_prob_confusion(word, prob):
	word_upper = word.title()
	return get_value(word, prob) + get_value(word_upper, prob)

# prob : unigram probability dict gathered from corpus
def spell_check_unigram(word_list, prob):
	# total number of corrections made
	result = []
	for i in range(len(word_list)):
		word = word_list[i]
		# 0 for non-alpha, 1 for lowercase, 2 for containing uppercase
		word_class = word_classification(word)
		# continue if not alpha
		if word_class == 0:
			result.append(word)
			continue
		# word candidate to be appended
		cand_word = word
		word_lower = word.lower()
		# probability for next word
		# Need this for 3-way cases (went/want, want/wont)
		prob_cand_word = 0
		for w1, w2 in confusion_set:
			if w1 == word_lower or w2 == word_lower:
				w1_prob = get_prob_confusion(w1, prob)
				#print w1
				#print w1_prob
				w2_prob = get_prob_confusion(w2, prob)
				#print w2
				#print w2_prob
				# if w1 and w2 have the same or better prob, leave as is
				if w1_prob == w2_prob:
					prob_cand_word = w1_prob
					continue
				# if w2 has better prob, change it to w2
				elif w1_prob < w2_prob and w2_prob > prob_cand_word:
					# if the word being checked was already lowercase
					prob_cand_word = w2_prob
					if word_class == 1:
						cand_word = w2
					# Case: Uppercase
					else:
						cand_word = w2.title()
				# w1 with better prob
				elif w1_prob > w2_prob and w1_prob > prob_cand_word:
					prob_cand_word = w1_prob
					if word_class == 1:
						cand_word = w1
					#case : uppercase
					else:
						cand_word = w1.title()
		result.append(cand_word)
	#print word_list
	return result

########################################################################
########################################################################
########################################################################
#######                       BIGRAMS                            #######

# UNSMOOTHED BIGRAM FIRST


def generate_bigrams(corpus_sentence):
	bigram_list = []
	unigram_list = []
	sentence_with_boundary = generate_sentence_markers(corpus_sentence)
	for sentence in sentence_with_boundary:
		word_list = sentence.split(' ')
		for i in range(1, len(word_list)):
			# append tuple
			bigram_list.append((word_list[i-1], word_list[i]))
		for i in range(0, len(word_list)):
			unigram_list.append(word_list[i])
	#print bigram_list
	#print unigram_list
	bigram_counter = collections.Counter(bigram_list)
	unigram_counter = collections.Counter(unigram_list)
	bigram_prob = {}
	for key in bigram_counter:
		bigram_prob[key] = bigram_counter[key] / float(unigram_counter[key[0]])
	#print bigram_prob
	return bigram_list, bigram_counter, bigram_prob

# modified sentence boundary
def bigram_prob_word(prob, word, prev, next):
	#print 'AAAAAAA'
	#print 'BIGGGRAAAM PROB'
	#print prev + ' ' + word + ' ' + next
	word_upper = word.title()
	prob_word_given_prev = get_value((prev, word), prob) + get_value((prev,word_upper), prob)
	prob_next_given_word = get_value((word,next), prob) + get_value((prev,word_upper), prob)
	#print prob_word_given_prev
	#print prob_next_given_word
	return prob_word_given_prev, prob_next_given_word

def sentence_to_word_list(sentences):
	result = []
	for sentence in sentences:
		word_list = sentence.split(' ')
		for word in word_list:
			result.append(word)
	return result

def spell_check_bigram(sentence_boundary, prob):
	result = []
	#print sentence_boundary
	word_list = sentence_to_word_list(sentence_boundary)
	#print word_list
	for i in range(len(word_list)):
		word = word_list[i]
		word_class = word_classification(word)
		# non alpha
		if(word_class == 0):
			if(word == '<s>' or word == '</s>'):
				continue
			result.append(word)
			continue
		word_lower = word.lower()
		# candidate word
		cand_word = word
		prob_cand_word = 0
		for w1, w2 in confusion_set:
			if w1 == word_lower or w2 == word_lower:
				# invariant : there 
				w1_prob_prev, w1_prob_next = bigram_prob_word(prob, w1, word_list[i-1], word_list[i+1])
				w2_prob_prev, w2_prob_next = bigram_prob_word(prob, w2, word_list[i-1], word_list[i+1])
				w1_prob = w1_prob_prev * w1_prob_next
				w2_prob = w2_prob_prev * w2_prob_next
				# if w1 and w2 have the same or better prob, leave as is
				if w1_prob == w2_prob:
					prob_cand_word = w1_prob
					# if probability is 0, take a look at both prev and next prob, select one with higher
					# For unsmoothed bigram
					if(w1_prob == 0):
						sum_w1_prob = w1_prob_prev + w1_prob_next
						sum_w2_prob = w2_prob_prev + w2_prob_next
						if(sum_w1_prob == sum_w2_prob):
							continue
						if(sum_w1_prob > sum_w2_prob):
							cand_word = w1
						if(sum_w1_prob < sum_w2_prob):
							cand_word = w2
					continue
				# if w2 has better prob, change it to w2
				elif w1_prob < w2_prob and w2_prob > prob_cand_word:
					# if the word being checked was already lowercase
					prob_cand_word = w2_prob
					if word_class == 1:
						cand_word = w2
					# Case: Uppercase
					else:
						cand_word = w2.title()
				# w1 with better prob
				elif w1_prob > w2_prob and w1_prob > prob_cand_word:
					prob_cand_word = w1_prob
					if word_class == 1:
						cand_word = w1
					#case : uppercase
					else:
						cand_word = w1.title()
		result.append(cand_word)
	#print word_list
	return result

########################################################################
########################################################################
########################################################################
#######               BIGRAMS  WITH SMOOTHING                    #######

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






########################################################################
########################################################################
####   PROCESSING, AND TAKING CARE OF IO FOR TEXT TO BE CHECKED   ######

# returns number of corrections that should have been made, but not
def compare_texts (mod, train):
	mod_conf_set = []
	train_conf_set = []
	#print mod
	for word in mod:
		word_lower = word.lower()
		#print word
		for w1, w2 in confusion_set:
			if word_lower == w1 or word_lower == w2:
				mod_conf_set.append(word_lower)
				# need to break; there can be multiples
				break
	for word in train:
		word_lower = word.lower()
		for w1, w2 in confusion_set:
			if word_lower == w1 or word_lower == w2:
				train_conf_set.append(word_lower)
				# need to break; there can be multiples
				break
	#print mod_conf_set
	#print train_conf_set
	mod_counter = collections.Counter(mod_conf_set)
	train_counter = collections.Counter(train_conf_set)
	#print mod_counter
	#print train_counter
	# number of errors that haven't been corrected
	diff = mod_counter - train_counter
	sum_diff = sum(diff.values())
	# print sum_diff
	return sum_diff

# replace back to may be
def post_process(result_list):
	text = ' '.join(result_list)
	text = text.replace('mayxyzbe', 'may be')
	text = text.replace('Mayxyzbe', 'May be')
	return text.split(' ')

def run_spell_check_test (corpus_name, prob):
	path_read = "./data_corrected/spell_checking_task_v2/%s/test_modified_docs/*.txt" % corpus_name
	dir_write = "./data_corrected/spell_checking_task_v2/%s/test_docs/" % corpus_name

	for filename in glob.glob(path_read):
		modified_sentence = []
		modified_words = []
		with open(filename, 'r') as f:
			for line in f:
				line = line.replace('may be', 'mayxyzbe')
				line = line.replace('May be', 'Mayxyzbe')
				modified_sentence += nltk.tokenize.sent_tokenize(line)
				modified_words += line.split(' ')
		modified_sentence_boundary = generate_sentence_markers(modified_sentence)
		corrected_words = spell_check_bigram(modified_sentence_boundary, prob)
		corrected_words = post_process(corrected_words)

		temp = filename.split('/')
		file_path_w = temp[len(temp)-1]
		file_path_w = dir_write + '/' + file_path_w
		f = open(file_path_w, 'w')
		f.write(' '.join(corrected_words))
	return


def run_spell_check_development (corpus_name, unigram_prob, bigram_prob, bigram_smooth_counter, total_doc_number,train_length):
	result_spell_check = []
	path_mod = "./data_corrected/spell_checking_task_v2/%s/train_modified_docs/*.txt" % corpus_name
	# CHANGE 
	for i in range(train_length, total_doc_number):
	#for i in range(total_doc_number):
		filename_modified = glob.glob(path_mod)[i]
		# filename for training files
		filename_train = filename_modified.replace('_modified', '')
		#print filename_modified
		#print filename_train
		modified_sentence = []
		modified_words = []
		modified_corrected_words = []
		train_words = []
		#print filename
		with open(filename_modified, 'r') as f:
			for line in f:
				line = line.replace('may be', 'mayxyzbe')
				line = line.replace('May be', 'Mayxyzbe')
				modified_sentence += nltk.tokenize.sent_tokenize(line)
				modified_words += line.split(' ')
			#print text_words
		with open(filename_train, 'r') as f:
			for line in f:
				train_words += line.split(' ')
		unigram_corrected_words = spell_check_unigram(modified_words, unigram_prob)
		modified_sentence_boundary = generate_sentence_markers(modified_sentence)
		bigram_corrected_words = spell_check_bigram(modified_sentence_boundary, bigram_prob)
		bigram_smooth_corrected_words = spell_check_bigram(modified_sentence_boundary, bigram_smooth_counter)

		unigram_corrected_words = post_process(unigram_corrected_words)
		bigram_corrected_words = post_process(bigram_corrected_words)
		bigram_smooth_corrected_words = post_process(bigram_smooth_corrected_words)


		#print train_words
		#print unigram_corrected_words
		num_uncorrected_unigram = compare_texts(unigram_corrected_words, train_words)
		num_uncorrected_bigram = compare_texts(bigram_corrected_words, train_words)
		num_uncorrected_bigram_smooth = compare_texts(bigram_smooth_corrected_words, train_words)
		#print num_uncorrected
		result_spell_check.append((filename_modified, bigram_smooth_corrected_words, num_uncorrected_bigram_smooth,\
		bigram_corrected_words, num_uncorrected_bigram, unigram_corrected_words, num_uncorrected_unigram))
	return result_spell_check

def analyze_and_write_development (result_spell_check, corpus_name):
	write_path_uni = "./data_corrected/spell_checking_task_v2/%s_result_unigram.txt" % corpus_name
	write_path_bi = "./data_corrected/spell_checking_task_v2/%s_result_bigram.txt" % corpus_name
	write_path_s = "./data_corrected/spell_checking_task_v2/%s_result_bigram_smooth.txt" % corpus_name

	directory = "./data_corrected/spell_checking_task_v2/%s/checked_modified_docs" % corpus_name
	# make the folder
	try: 
		os.makedirs(directory)
	except OSError:
		if not os.path.isdir(directory):
			raise
	sum_diff_uni = 0
	sum_diff_bi = 0
	sum_diff_s = 0
	f_uni = open(write_path_uni, 'w')
	f_bi = open(write_path_bi, 'w')
	f_s = open(write_path_s, 'w')
	lines_uni = []
	lines_bi = []
	lines_s = []
	for name, text_s, num_s,text_bi, num_bi, text_uni, num_uni in result_spell_check:
		temp = name.split('/')
		filename = temp[len(temp)-1]

		lines_uni.append(filename + ' ' + str(num_uni) + '\n')
		lines_bi.append(filename + ' ' + str(num_bi) + '\n')
		lines_s.append(filename + ' ' + str(num_s) + '\n')

		sum_diff_uni += num_uni
		sum_diff_bi += num_bi
		sum_diff_s += num_s

		text_directory = directory + '/' + filename
		f2 = open(text_directory, 'w')
		f2.write(' '.join(text_bi))
	f_uni.write('TOTAL NUM ERRORS:' + str(sum_diff_uni) + '\n')
	f_bi.write('TOTAL NUM ERRORS:' + str(sum_diff_bi) + '\n')
	f_s.write('TOTAL NUM ERRORS:' + str(sum_diff_s) + '\n')
	for l in lines_uni:
		f_uni.write(l)
	for l in lines_bi:
		f_bi.write(l)
	for l in lines_s:
		f_s.write(l)

def run():
	for corpus in corpus_names:
		print 'checking ' + corpus + '...'
		sentence, word, total_num, train_num = preprocess(corpus)
		unigram, unigram_counter, unigram_prob = generate_unigrams(word)
		bigram, bigram_counter, bigram_prob = generate_bigrams(sentence)
		#print unigram_counter
		#print count_of_counts(unigram_counter, unigram, 'unigram')

		bigram_smooth_counter = good_turing(count_of_counts(bigram_counter, bigram, 'bigram'), bigram_counter, 'bigram')

		unigram_smooth_counter = good_turing(count_of_counts(unigram_counter, unigram, 'unigram'), unigram_counter, 'unigram')
		
		# bigram_smooth_counter can be used directly; it is only looking for relative probability
		# counter and prob can be used interchangeably
		result = run_spell_check_development(corpus, unigram_prob, bigram_prob, bigram_smooth_counter, total_num, train_num)
		analyze_and_write_development(result, corpus)
		# USE BIGRAM MODEL AS IT WORKS BEST
		run_spell_check_test(corpus, bigram_prob)
run()