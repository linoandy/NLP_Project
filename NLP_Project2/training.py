#coding=utf-8 

###############
# please unzip the data file, and put the folder nlp_project2_uncertainty 
# in the same location as this script, training.py
# then run training.py and you should get the baseline output on screen and in csv
###############

import glob
import re
import nltk
import csv


def BIO_tagger(file): # this function processes the document passed in, and replace CUE tags with BIO tags
	token_lists = []
	with open(file, 'r') as f:
		for line in f:
			# split by tab
			if len(line.split('\t')) == 3: # filter out the very last line of each document, usually it's '\n'
				token_lists.append(line.split('\t'))

		# replace '_\n' with tag 'O'
		for token_list in token_lists:
			if token_list[2] == '_\n':
				token_list[2] = 'O'

		previous_tagger = ''
		for i in range(len(token_lists)):
			previous_tagger_temp = token_lists[i][2]
			if token_lists[i][2].find('CUE') != -1:
				# replace 'CUE-n' that don't equal to the tag of 'CUE-(n-1)' with 'B-CUE' 
				if i == 0 or token_lists[i][2] != previous_tagger:
					token_lists[i][2] = 'B-CUE'
				# replace other 'CUE-n' with 'I-CUE'
				else:
					token_lists[i][2] = 'I-CUE'
			previous_tagger = previous_tagger_temp
	return token_lists

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
# 	# play with it in nltk tagger
# 	nltk_pos_tag_result = nltk.pos_tag(sentence)
# 	error = 0
# 	for i in range(len(list_of_tokens)):
# 		if list_of_tokens[i][1] != nltk_pos_tag_result[i][1]:
# 			error += 1
# 	print filename, ' error: ', error


# baseline system
def baseline_dict(path):
	baseline_list = []
	baseline_set = []
	for file_name in glob.glob(path):
		training_set_threshold = len(glob.glob(path)) * 0.01
		if int(re.findall('[0-9]+', file_name)[1]) < training_set_threshold:
			baseline_set = BIO_tagger(file_name)
		for baseline in baseline_set:
			if (baseline[2] == 'B-CUE' or baseline[2] == 'I-CUE') and baseline[0] not in baseline_list:
				baseline_list.append(baseline[0])
	return baseline_list

def baseline_calculation(file, baseline_data_set):
	token_lists = []
	baseline_result = {}
	with open(file, 'r') as f:
		for line in f:
			# split by tab
			if len(line.split('	')) == 3: # filter out the very last line of each document, usually it's '\n'
				token_lists.append(line.split('	'))

		correct = 0
		num_of_prediction = float(len(token_lists))
		for token_list in token_lists:
			if token_list[0] in baseline_data_set and token_list[2].find('CUE') != -1:
				correct += 1
		baseline_result['file'] = file
		baseline_result['precision'] = correct / num_of_prediction
		baseline_result['recall'] = correct / num_of_prediction
	return baseline_result

baseline_dictionary = baseline_dict("./nlp_project2_uncertainty/train/*.txt")
for filename in development_set:
	print baseline_calculation(filename, baseline_dictionary)

test_public_path = './nlp_project2_uncertainty/test-public/*.txt'
test_private_path = './nlp_project2_uncertainty/test-private/*.txt'

def test_baseline(path, baseline_data_set):
	word_result = []
	sentence_result = []
	word_num = 0
	sentence_num = 0
	for filename in glob.glob(path):
		with open(filename, 'r') as f:
			sentence_bool = 0
			prev_line_blank = False
			for line in f:
				words = line.split()

				if len(words) == 0:
					if(prev_line_blank == True):
						continue
					prev_line_blank = True
					sentence_num += 1
					continue
				else:
					prev_line_blank = False

				if words[0] in baseline_data_set:
					word_result.append(word_num)
					sentence_result.append(sentence_num)
				word_num += 1
	sentence_result = list(set(sentence_result))
	print sentence_num
	return word_result, sentence_result

def write_to_csv(word_result_pu, word_result_pr, sentence_result_pu, sentence_result_pr):
	def syntax_word(result):
		l = []
		s = ''
		for r in result:
			s += str(r) + '-' + str(r) +' '
		l.append(s)
		return l
	def syntax_sentence(result):
		l = []
		s = ''
		for r in result:
			s += str(r) + ' '
		l.append(s)
		return l
	w_pu = syntax_word(word_result_pu)
	w_pr = syntax_word(word_result_pr)
	s_pu = syntax_sentence(sentence_result_pu)
	s_pr = syntax_sentence(sentence_result_pr)
	with open('word_result_baseline.csv', 'wb') as f:
		a = csv.writer(f)
		a.writerow(['Type', 'Spans'])
		a.writerow(['CUE-public'] + w_pu)
		a.writerow(['CUE-private'] + w_pr)
	with open('sentence_result_baseline.csv', 'wb') as f:
		a = csv.writer(f)
		a.writerow(['Type', 'Indices'])
		a.writerow(['SENTENCE-public'] + s_pu)
		a.writerow(['SENTENCE-private'] + s_pr)
	return

word_result_pu, sentence_result_pu = test_baseline(test_public_path, baseline_dictionary)
word_result_pr, sentence_result_pr = test_baseline(test_private_path, baseline_dictionary)
write_to_csv(word_result_pu, word_result_pr, sentence_result_pu, sentence_result_pr)
print baseline_dictionary


