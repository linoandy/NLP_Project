import glob
import re
from threading import Thread
import gensim
import json

WORD2VEC = './brown_model'
model = gensim.models.Word2Vec.load('./brown_model')


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
		# for the purpose of this project, this time we save all data into training data set, so we set threshold to 1
	    training_set_threshold = len(glob.glob(path)) * 1
	    if int(re.findall('[0-9]+', file_name)[1]) < training_set_threshold:
	        training_set.append(file_name)
	    else:
	        development_set.append(file_name)
	return training_set, development_set

def uncertain_word():
	training_data, development_data = breakup_data()
	# this list contains all words in corpus
	words = []
	for filename in training_data:
		list_of_tokens = BIO_tagger(filename)
		for token in list_of_tokens:
			if (token[2] == 'I-CUE' or token[2] == 'B-CUE') and token[0] not in words:
				words.append((token[0].lower(), 1))
			elif token[0] not in words:
				words.append((token[0].lower(), -1))
	return words

def main():
	#wordlist = uncertain_word()
	wordlist = []
	uncertain_set = []

	with open('word_list.txt', 'r') as f:
		for line in f:
			wordlist.append(line.strip())
	with open('uncertain_list.txt', 'r') as f:
		for line in f:
			uncertain_set.append(line.strip())

	print wordlist
	#k = 3 # change k  here
	#kwv = knn.KnnWordVec(wordlist, k)
	print "start evaluating words"
	dictionary = {}

	def result(target_word, model, main_dictionary):
		main_dictionary[target_word] = model.knn_run(target_word)
		print target_word, main_dictionary[target_word]

	list_threads = []
	n = 1
	certain_set = []
	#for w in wordlist:
	#	if(w[1] == 1):
	#		uncertain_set.append(w[0])

	def categorize(word):
		for sw in synset:
			try:
				sim_score = model.similarity(word, sw)
				if(sim_score > .9):
					return 1
			except:
				continue
			
		return -1
	expanded_set = []

	def categorize2():

		for w in uncertain_set:
			try:
				most_sim = model.most_similar(positive=[w])
				#print most_sim[0][0]
				expanded_set.append((w, most_sim[0][0], most_sim[1][0]))
				print most_sim[0][0]
			except:
				continue
		return expanded_set


	ex_s = categorize2()
	for s in expanded_set:
		print s[0] + ' ' + s[1] + ' ' + s[2]
	for s in expanded_set:
		dictionary[s[0]] = 1
		dictionary[s[1]] = 1
		dictionary[s[2]] = 1





		#dictionary[word[0]] = categorize(word[0])
		#print word, dictionary[word[0]]
		# t = Thread(target=result, args=(word[0], kwv, dictionary))
		# list_threads.append(t)
		# t.start()
		# if len(list_threads) >= 5 or n == len(wordlist):
		# 	for z in list_threads:
		# 		z.join()
		# 	list_threads = []
		# n += 1

	with open('word_embedding.json', 'w') as f:
		f.write(json.dumps(dictionary, ensure_ascii=False))
	with open('expanded_set.txt', 'w') as f:
		for s in expanded_set:
			f.write(s[0] + ' ' + s[1] + ' ' + s[2] + '\n')

	return dictionary

main()



