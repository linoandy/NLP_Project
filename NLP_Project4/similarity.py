from fuzzywuzzy import fuzz
import glob
import re

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

def breakup_data():
	# breaking up the data set
	# print '\n\n\nbreaking up the data set into training and development in progress...'
	training_set = []
	development_set = []
	path = "./nlp_project2_uncertainty/train/*.txt"
	for file_name in glob.glob(path):
	    training_set_threshold = len(glob.glob(path)) * 0.9
	    if int(re.findall('[0-9]+', file_name)[1]) < training_set_threshold:
	        training_set.append(file_name)
	    else:
	        development_set.append(file_name)
	return training_set, development_set

def uncertain_word():
	training_data, development_data = breakup_data()
	words = []
	for filename in training_data:
		list_of_tokens = BIO_tagger(filename)
		for token in list_of_tokens:
			if (token[2] == 'I-CUE' or token[2] == 'B-CUE') and token[0] not in words:
				words.append(token[0].lower())
	return words

def calculation(test_word, uncertain_word_list):
	# # break up data set into training set and development set
	# # save all CUE words in training set to a list
	# global uncertain_word_list
	# if len(uncertain_word_list) == 0:
	# 	uncertain_word_list = uncertain_word()
	# calculate fuzzy ratio of the target word
	total_score = 0.0
	total_score_possible = 100.0 * len(uncertain_word_list)
	for word in uncertain_word_list:
		total_score += fuzz.ratio(test_word.lower(), word)
	similar_ratio = total_score / total_score_possible	
	# determine if the target word is similar to CUE word
	if similar_ratio > 0.19:
		is_similar = True
	else:
		is_similar = False
	return is_similar

# # break up data set into training set and development set
# # save all CUE words in training set to a list
# global uncertain_word_list
# if len(uncertain_word_list) == 0:
# 	uncertain_word_list = uncertain_word()
# # uncertain_word_list = []

