#coding=utf-8 

import glob
import re
import nltk

def BIO_tagger(file): # this function processes the document passed in, and replace CUE tags with BIO tags
	token_lists = []
	with open(file, 'r') as f:
		for line in f:
			# split by tab
			if len(line.split('	')) == 3: # filter out the very last line of each document, usually it's '\n'
				token_lists.append(line.split('	'))

		# replace '_\n' with tag 'O'
		for token_list in token_lists:
			if token_list[2] == '_\n':
				token_list[2] = 'O'

		for i in range(len(token_lists)):
			if token_lists[i][2].find('CUE') != -1:
				# replace 'CUE-n' directly after tag 'O' with 'B-CUE' 
				if i == 0 or token_lists[i-1][2] == 'O':
					token_lists[i][2] = 'B-CUE'
				# replace other 'CUE-n' with 'I-CUE'
				else:
					token_lists[i][2] = 'I-CUE'
	return token_lists

# baseline system
baseline_dictionary = []
path = "./nlp_project2_uncertainty/train/*.txt"
for file_name in glob.glob(path):
	training_set_threshold = len(glob.glob(path)) * 0.01
	if int(re.findall('[0-9]+', file_name)[1]) < training_set_threshold:
		baseline_set = BIO_tagger(file_name)
	for baseline in baseline_set:
		if (baseline[2] == 'B-CUE' or baseline[2] == 'I-CUE') and baseline[0] not in baseline_dictionary:
			baseline_dictionary.append(baseline[0])
print baseline_dictionary


# breaking up the data set
training_set = []
development_set = []
path = "./nlp_project2_uncertainty/train/*.txt"
for file_name in glob.glob(path):
	training_set_threshold = len(glob.glob(path)) * 0.9
	if int(re.findall('[0-9]+', file_name)[1]) < training_set_threshold:
		training_set.append(file_name)
	else:
		development_set.append(file_name)

# process training set by BIO tagging and concatenate tokens into sentence
for filename in training_set:
	sentence = []
	list_of_tokens = BIO_tagger(filename)
	for token in list_of_tokens:
		sentence.append(token[0])

	# play with it in nltk tagger
	nltk_pos_tag_result = nltk.pos_tag(sentence)
	error = 0
	for i in range(len(list_of_tokens)):
		if list_of_tokens[i][1] != nltk_pos_tag_result[i][1]:
			error += 1
	print filename, ' error: ', error


