#coding=utf-8 

###############
# please unzip the data file, and put the folder nlp_project2_uncertainty 
# in the same location as this script, single_feature_model.py
# then run single_feature_model.py and you should get the output on screen and in csv
###############

import glob
import re
import nltk
import csv
import random
from nltk.metrics.scores import f_measure
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
from itertools import chain

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
print '\n\n\nbreaking up the data set into training and development in progress...'
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
	formatted_pos_tag = []
	for filename in dataset:
		sentence = []
		pos_tag = []
		list_of_tokens = BIO_tagger(filename)
		for token in list_of_tokens:
			# for some reason, we MUST use text in unicode
			# sentence.append((token[0].lower().decode('utf-8'), token[2]))
			# replace I-tag with B-tag, using only BO tags
			tag = 'B-CUE' if token[2] == 'I-CUE' else token[2]
			# word_token = token[0].lower().decode('utf-8') if token[0] not in single_occurance_word else '<UNK>'.decode('utf-8')
			sentence.append((token[0].decode('utf-8').lower(), tag))
			# sentence.append((word_token, token[2]))
			pos_tag.append((token[1].decode('utf-8'), token[2]))
		formatted_sentence.append(sentence)
		formatted_pos_tag.append(pos_tag)
	# print len(formatted_sentence), len(formatted_pos_tag)
	return formatted_sentence, formatted_pos_tag

def data_selector_formater(dataset):
	formatted_sentence = []
	formatted_pos_tag = []
	for filename in dataset:
		sentence = []
		pos_tag = []
		selector_flag = []
		list_of_tokens = BIO_tagger(filename)
		for token in list_of_tokens:
			# for some reason, we MUST use text in unicode
			# sentence.append((token[0].lower().decode('utf-8'), token[2]))
			selector_flag.append(token[2])
			# replace I-tag with B-tag, using only BO tags
			tag = 'B-CUE' if token[2] == 'I-CUE' else token[2]
			sentence.append((token[0].decode('utf-8').lower(), token[2]))
			pos_tag.append((token[1].decode('utf-8'), token[2])) 
		# throw away files that don't have 'B-CUE' or 'I-CUE'
		if 'B-CUE' in selector_flag or 'I-CUE' in selector_flag:
			formatted_sentence.append(sentence)
			formatted_pos_tag.append(pos_tag)
		if 'B-CUE' not in selector_flag and 'I-CUE' not in selector_flag and random.random() >= 1.1:
			formatted_sentence.append(sentence)
			formatted_pos_tag.append(pos_tag)
	# print len(formatted_sentence), len(formatted_pos_tag)
	return formatted_sentence, formatted_pos_tag

def counter(dataset):
	corpus = []
	for filename in dataset:
		list_of_tokens = BIO_tagger(filename)
		for token in list_of_tokens:
			corpus.append(token[0])
	counter_temp = nltk.FreqDist(corpus)
	for word_count in counter_temp:
		if counter_temp[word_count] == 1:
			corpus.append(word_count)
	return corpus

single_occurance_word = counter(training_set)

# process training set by BIO tagging and concatenate tokens into sentence
print '\n\n\nformatting training data set in progress...'
training_sentence, training_pos_tag = data_formater(training_set)

# process development set by BIO tagging and concatenate tokens into sentence
print '\n\n\nformatting development data set in progress...'
development_sentence, development_pos_tag = data_formater(development_set)
evaluation_set = development_sentence

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) #- {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

# evaluate
def evaluation (correct_sent_set, model):
	correct_sentence_set = list(correct_sent_set)
	sentence_to_evaluate = []
	for correct_sentence in correct_sentence_set:
		sentence_temp = []
		for correct_token in correct_sentence:
			sentence_temp.append(correct_token[0])
		sentence_to_evaluate.append(sentence_temp)

	if model == 'crf':
		# make prediction using crf model
		prediction_word = ct.tag_sents(sentence_to_evaluate)
		# prediction_pos = ct_pos.tag_sents(test_pos_set)
	elif model == 'hmm':
		# make prediction using hmm model
		prediction_word = []
		prediction_pos = []
		for test_sentence in sentence_to_evaluate:
			prediction_word.append(tagger.tag(test_sentence))
		# for test_pos in test_pos_set:
		# 	prediction_pos.append(tagger.tag(test_pos))
	elif model == 'perceptron':
		# make prediction using perceptron model
		prediction_word = []
		prediction_pos = []
		for test_sentence in sentence_to_evaluate:
			prediction_word.append(pt.tag(test_sentence))
		# for test_pos in test_pos_set:
		# 	prediction_pos.append(tagger.tag(test_pos))

	correct_set = []
	# print correct_sentence_set
	for evaluate_result in correct_sentence_set:
		correct_set_temp = []
		for i in range(len(evaluate_result)):
			correct_set_temp.append(evaluate_result[i][1])
		correct_set.append(correct_set_temp)
		# correct_set_temp += evaluate_result

	prediction_to_evaluate = []
	for evaluate_result in prediction_word:
		prediction_to_evaluate_temp = []
		for i in range(len(evaluate_result)):
			prediction_to_evaluate_temp.append(evaluate_result[i][1])
		prediction_to_evaluate.append(prediction_to_evaluate_temp)

	# for a in range(len(correct_set_temp)):
	# 	correct_set.append(correct_set_temp[a][1])
	# 	prediction_to_evaluate.append(prediction_to_evaluate_temp[a][1])
	# print len(correct_set), len(prediction_to_evaluate)
	# print model, 'f_measure', f_measure(set(prediction_to_evaluate), set(correct_set), alpha=0.5)
	print model.upper(), 'f_measure'
	print bio_classification_report(correct_set, prediction_to_evaluate)
	return

# train and evaluate nltk tagging crf module
print '\n\n\ntraining crf model in progress...'
ct = nltk.tag.CRFTagger()
ct_pos = nltk.tag.CRFTagger()
ct.train(training_sentence,'model.crf.tagger')
ct_pos.train(training_pos_tag, 'model.crf_pos.tagger')
ct.set_model_file('model.crf.tagger')
print "\n\n\nevaluation of crf model: %.3f%%" % (100 * ct.evaluate(development_sentence))

# train and evaluate nltk tagging hmm module
print '\n\n\ntraining hmm model in progress...'
ht = nltk.tag.hmm.HiddenMarkovModelTrainer()
tagger = ht.train_supervised(training_sentence)
tagger_pos = ht.train_supervised(training_pos_tag)
print "\n\n\nevaluation of hmm model: %.3f%%" % (100 * tagger.evaluate(development_sentence))

# train and evaluate nltk tagging perceptron module
print '\n\n\ntraining perceptron model in progress...'
pt = nltk.tag.perceptron.PerceptronTagger(load=False)
pt_pos = nltk.tag.perceptron.PerceptronTagger(load=False)
pt.train(training_sentence, 'model.perceptron.tagger', nr_iter=8)
pt_pos.train(training_pos_tag, 'model.perceptron_pos.tagger', nr_iter=8)
print "\n\n\nevaluation of perceptron model: %.3f%%" % (100 * pt.evaluate(development_sentence))

evaluation(development_sentence, 'crf')
evaluation(development_sentence, 'hmm')
evaluation(development_sentence, 'perceptron')

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

def test_model(path, model):
	word_result = []
	sentence_result = []
	word_num = 0
	sentence_num = 0
	test_sentence_set = []
	test_pos_set = []
	for filename in glob.glob(path):
		with open(filename, 'r') as f:
			test_sentence = []
			test_pos = []
			prev_line_blank = False
			for line in f:
				words = line.split()

				if len(words) == 0:
					if(prev_line_blank == True):
						continue
					prev_line_blank = True
					test_sentence_set.append(test_sentence)
					test_pos_set.append(test_pos)
					test_sentence = []
					test_pos = []
					continue
				else:
					prev_line_blank = False
					test_sentence.append(words[0].decode('utf-8').lower())
					test_pos.append(words[1].decode('utf-8').lower())

	if model == 'crf':
		# make prediction using crf model
		prediction_word = ct.tag_sents(test_sentence_set)
		prediction_pos = ct_pos.tag_sents(test_pos_set)
	elif model == 'hmm':
		# make prediction using hmm model
		prediction_word = []
		prediction_pos = []
		for test_sentence in test_sentence_set:
			prediction_word.append(tagger.tag(test_sentence))
		for test_pos in test_pos_set:
			prediction_pos.append(tagger.tag(test_pos))
	elif model == 'perceptron':
		# make prediction using perceptron model
		prediction_word = []
		prediction_pos = []
		for test_sentence in test_sentence_set:
			prediction_word.append(pt.tag(test_sentence))
		for test_pos in test_pos_set:
			prediction_pos.append(tagger.tag(test_pos))

	for j in range(len(prediction_word)):
		for k in range(len(prediction_word[j])):
			if prediction_word[j][k][1] != prediction_pos[j][k][1] and prediction_word[j][k][0] == prediction_pos[j][k][0]:
				# print prediction_word[j][k], prediction_pos[j][k]
				lst = list(prediction_word[j][k])
				lst[1] = 'O'
				prediction_word[j][k] = tuple(lst)
	
	for single_sentence in prediction_word:
		for single_token in single_sentence:
			# print single_token
			if single_token[1] == 'B-CUE' or single_token[1] == 'I-CUE':
				word_result.append(word_num)
				sentence_result.append(sentence_num)
			word_num += 1
		sentence_num += 1

	print word_num, sentence_num
	return word_result, sentence_result


def write_to_csv(word_result_pu, word_result_pr, sentence_result_pu, sentence_result_pr):
	def syntax_word(result):
		l = []
		s = ''
		temp_list = []
		if len(result) == 0:
			return l
		temp_list.append(result[0])
		for i in range(1, len(result)):
			if result[i] == result[i-1]+1:
				if len(temp_list) == 0:
					temp_list.append(result[i-1])
					temp_list.append(result[i])
				else:
					temp_list.append(result[i])
			# no consecutive result AND there are some in temp list to be flushed
			elif len(temp_list) != 0:
				s += str(temp_list[0]) + '-' + str(temp_list[(len(temp_list)-1)]) + ' '
				temp_list = []
		l.append(s)
		return l
	def syntax_sentence(result):
		l = []
		s = ''
		l_temp = []
		for r in result:
			if r not in l_temp:
				s += str(r) + ' '
				l_temp.append(r)
		l.append(s)
		return l
	w_pu = syntax_word(word_result_pu)
	w_pr = syntax_word(word_result_pr)
	s_pu = syntax_sentence(sentence_result_pu)
	s_pr = syntax_sentence(sentence_result_pr)
	with open('word_result.csv', 'wb') as f:
		a = csv.writer(f)
		a.writerow(['Type', 'Spans'])
		a.writerow(['CUE-public'] + w_pu)
		a.writerow(['CUE-private'] + w_pr)
	with open('sentence_result.csv', 'wb') as f:
		a = csv.writer(f)
		a.writerow(['Type', 'Indices'])
		a.writerow(['SENTENCE-public'] + s_pu)
		a.writerow(['SENTENCE-private'] + s_pr)
	return
word_result_pu, sentence_result_pu = test_model(test_public_path, 'crf')
word_result_pr, sentence_result_pr = test_model(test_private_path, 'crf')
# print word_result_pu
# word_result_pu, sentence_result_pu = test_baseline(test_public_path, baseline_dictionary)
# word_result_pr, sentence_result_pr = test_baseline(test_private_path, baseline_dictionary)
write_to_csv(word_result_pu, word_result_pr, sentence_result_pu, sentence_result_pr)
# SENTENCE-privatent baseline_dictionary
