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
print 'breaking up the data set into training and development in progress...\n\n\n'
training_set = []
development_set = []
path = "./nlp_project2_uncertainty/train/*.txt"
for file_name in glob.glob(path):
	training_set_threshold = len(glob.glob(path)) * 0.9
	if int(re.findall('[0-9]+', file_name)[1]) < training_set_threshold:
		training_set.append(file_name)
	else:
		development_set.append(file_name)

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

# baseline_dictionary = baseline_dict("./nlp_project2_uncertainty/train/*.txt")
# for filename in development_set:
# 	print baseline_calculation(filename, baseline_dictionary)

def data_formater(dataset):
	formatted_sentence = []
	for filename in dataset:
		sentence = []
		list_of_tokens = BIO_tagger(filename)
		for token in list_of_tokens:
			# for some reason, we MUST use text in unicode 
			sentence.append((token[0].decode('utf-8'), token[2]))
		formatted_sentence.append(sentence)
	return formatted_sentence

# process training set by BIO tagging and concatenate tokens into sentence
print 'formatting training data set in progress...\n\n\n'
training_sentence = data_formater(training_set)

# process development set by BIO tagging and concatenate tokens into sentence
print 'formatting development data set in progress...\n\n\n'
development_sentence = data_formater(development_set)

# train and evaluate nltk tagging crf module
print 'training crf model in progress...\n\n\n'
ct = nltk.tag.CRFTagger()
ct.train(training_sentence,'model.crf.tagger')
ct.set_model_file('model.crf.tagger')
print "evaluation of crf model: %.3f%%\n\n\n" % (100 * ct.evaluate(development_sentence))

# train and evaluate nltk tagging hmm module
print 'training hmm model in progress...\n\n\n'
ht = nltk.tag.hmm.HiddenMarkovModelTrainer()
tagger = ht.train_supervised(training_sentence)
print "evaluation of hmm model: %.3f%%\n\n\n" % (100 * tagger.evaluate(development_sentence))

# train and evaluate nltk tagging perceptron module
print 'training perceptron model in progress...\n\n\n'
ht = nltk.tag.perceptron.PerceptronTagger(load=False)
ht.train(training_sentence, 'model.perceptron.tagger', nr_iter=5)
print "evaluation of perceptron model: %.3f%%\n\n\n" % (100 * ht.evaluate(development_sentence))

# test_public_path = './nlp_project2_uncertainty/test-public/*.txt'
# test_private_path = './nlp_project2_uncertainty/test-private/*.txt'
