import re
import glob
import nltk
import collections
import codecs
import os

corpus_names = ['atheism','autos','graphics','medicine','motorcycles','religion','space']

confusion_set = []
# getting rid of UTF-8 BOM at the beginning, converting to ascii
f = codecs.open('./data_corrected/spell_checking_task/confusion_set.txt', encoding='ascii', errors='ignore')
for line in f:
	words = line.split()
	for i in range(1):
		# remove end-of-line markers
		words[i].replace('\r\n', '')
	# insert confusion set as tuples
	confusion_set.append((words[0], words[1]))
print confusion_set

def preprocess(corpus_name):
	corpus_sentence = []
	corpus_word = []
	path_train = "./data_corrected/spell_checking_task/%s/train_docs/*.txt" % corpus_name
	for filename in glob.glob(path_train):
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
				corpus_sentence += nltk.tokenize.sent_tokenize(line)

				# build corpus of words
				corpus_word += line.split(' ')
	return (corpus_sentence, corpus_word)

def generate_sentence_markers(corpus_sentence):
	result = []
	for sentence in corpus_sentence:
		sentence = 	'<s> ' + sentence.encode('utf-8') + ' </s>'
		result.append(sentence)
	return result

########################################################################
########################################################################


def generate_unigrams(corpus_word):
	unigram_counter = collections.Counter(corpus_word)
	unigram_sum = sum(unigram_counter.values())
	unigram_probabilities = {}
	for key in unigram_counter:
		unigram_probabilities[key] = unigram_counter[key] / float(unigram_sum)
	return unigram_counter, unigram_probabilities

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
		# next word to be appended
		next_word = word
		word_lower = word.lower()
		# probability for next word
		# Need this for 3-way cases (went/want, want/wont)
		prob_next_word = 0
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
					prob_next_word = w1_prob
					continue
				# if w2 has better prob, change it to w2
				elif w1_prob < w2_prob and w2_prob > prob_next_word:
					# if the word being checked was already lowercase
					prob_next_word = w2_prob
					if word_class == 1:
						next_word = w2
					# Case: Uppercase
					else:
						next_word = w2.title()
				# w1 with better prob
				elif w1_prob > w2_prob and w1_prob > prob_next_word:
					prob_next_word = w1_prob
					if word_class == 1:
						next_word = w1
					#case : uppercase
					else:
						next_word = w1.title()
		result.append(next_word)
	#print word_list
	return result

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

def run_spell_check (corpus_name, prob):
	result_spell_check = []

	path_mod = "./data_corrected/spell_checking_task/%s/train_modified_docs/*.txt" % corpus_name
	for filename in glob.glob(path_mod):
		filename_modified = filename
		# filename for training files
		filename_train = filename.replace('_modified', '')
		#print filename_modified
		#print filename_train
		modified_sentence = []
		modified_words = []
		modified_corrected_words = []
		train_words = []
		#print filename
		with open(filename_modified, 'r') as f:
			for line in f:
				modified_sentence += nltk.tokenize.sent_tokenize(line)
				modified_words += line.split(' ')
				unigram_corrected_words = spell_check_unigram(modified_words, prob)
			#print text_words
			#print unigram_corrected
		with open(filename_train, 'r') as f:
			for line in f:
				train_words += line.split(' ')
		#print train_words
		#print unigram_corrected_words
		num_uncorrected = compare_texts(unigram_corrected_words, train_words)
		#print num_uncorrected
		result_spell_check.append((filename_modified, num_uncorrected))
	#print result_spell_check
	return result_spell_check

def analyze_and_write (result_spell_check, corpus_name):
	write_path = "./data_corrected/spell_checking_task/%s_result.txt" % corpus_name
	sum_diff = 0
	f = open(write_path, 'w')
	lines = []
	for name, num in result_spell_check:
		temp = name.split('/')
		filename = temp[len(temp)-1]
		lines.append(filename + ' ' + str(num) + '\n')
		sum_diff += num
	f.write('TOTAL NUM ERRORS:' + str(sum_diff) + '\n')
	for l in lines:
		f.write(l)



# TODO : loop through the corpora
# TODO : bigram 



temp_sentence, temp_word = preprocess('medicine')
temp_unigram, temp_unigram_probabilities = generate_unigrams(temp_word)

#print temp_unigram
#print temp_unigram_probabilities
# spell_check_unigram(temp_word, temp_unigram_probabilities)

result = run_spell_check('medicine', temp_unigram_probabilities)
analyze_and_write(result, 'medicine')


