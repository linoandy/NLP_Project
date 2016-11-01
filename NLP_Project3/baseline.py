import glob
import nltk
import re
'''
	PART 1. Question processing (list of keywords to IR)
		method 1: leave out the question word <- Method chosen for baseline
		method 2: get the noun phrases
		Baseline implementation : method 1 + removing stop words with nltk stopwords + removing punct
		Possible addition : stemming?
	PART 2. Question Classification
		Easy for our project, just use the first word in question
		Who, where, when. Can be expanded to more categories in Part 2
	TODO 3. Query Reformulation
		I was going to do this, but questionable if this is necessary..
		This is pretty minor. Might just leave it out.
			EX) When was laser invented
				=> Laser was invented in..
	PART 4. Passage Retrieval
		Since document retrieval was done already, just need to retrieve passages
		Extract sections, paragraphs, sentences from doc from retrived doc
		PART 4.1 DO NER on the retrived, and exclude any passages that don't include the answer type
		TODO 4.2 RANK the passages : LEFT it for part 2; not in baseline
			Num of named entities of right type 
			Question keywords <- Maybe only this?
			Longest exact sequence of question keywords..
			Rank of doc, etc.
		Baseline implementation : part 4.1. Order of passages just in order they were found
		(So basically from highest ranking to lowest ranking doc..)
	TODO 5. Answer Processing
		From the retrieved passage, retrieve answer
		Method 1 : Just return whatever matching entity
		Method 2 : Handwritten regex patterns? <- for part 2 maybe
			Pattern learning????
'''

starting_num = 89
ending_num = 320
current_num = starting_num

# constants for question types; need to be more robust for part 2
# baseline : WHO/WHERE/WHEN
WHO_TYPE = 0
WHERE_TYPE = 1
WHEN_TYPE = 2

q_path = "./question.txt"
a_path = "./answer.txt"
d_path = "./doc_dev"

###############################################################################
###############################################################################
# QUESTION PROCESSING
# Baseline implementation : method 1 + removing stop words with nltk stopwords + removing punct

# IN  : Question Description
# OUT : List of keywords,
#       just leaving out the question word and stop word list(METHOD 1)

# using a small stop word list; can use a bigger one later if wanted (experiment?)
# look here for the nltk stopwords used here, and some stopword code involved
# http://www.nltk.org/book/ch02.html

# reference : http://stackoverflow.com/a/32469562
stopwords = set(nltk.corpus.stopwords.words('english'))
# we don't want punctuation either
stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) 

def get_q_keywords(desc_token):
	l_stopped = [w for w in desc_token if w.lower() not in stopwords]
	return l_stopped

###############################################################################
###############################################################################
# QUESTION CLASSIFICATION
'''
BASELINE IMPLEMENTATION
	Baseline implementation : WHO/WHERE/WHEN
	Need to be more robust for part 2

IN  : Question Description
OUT : Question type
'''

def get_q_type(desc_token):
	first_word = desc_token[0]
	print first_word
	if(first_word == 'Who'):
		return WHO_TYPE
	if(first_word == 'Where'):
		return WHERE_TYPE
	if(first_word == 'When'):
		return WHEN_TYPE
	return -1

###############################################################################
###############################################################################
# PASSAGE RETRIEVAL
''' 
BASELINE IMPLEMENTATION
	DO NER on the retrived, and exclude any passages that don't include the answer type
	Currently using nltk's default NER; Using Stanford NER might work better, but have
	to download jar and shit. Might consider for part 2
	ORDER of the Passages retrieved : just the order they were found from docs 1..100
	Thus from the highest ranking doc to lowest ranking
TODO 4.2 RANK the passages
	Didn't do for baseline
'''
# IN  : Question number, Docs for the question
# OUT : List of passages


'''
From https://pythonprogramming.net/named-entity-recognition-nltk-tutorial/:
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
'''
# GH - I can't find the complete list of NER tags, but this should be enough for now

# WHO : PERSON
# WHERE : LOCATION, FACILITY, GPE
# WHEN : TIME, DATE
NER_TAG = [['PERSON'],['LOCATION', 'FACILITY', 'GPE'],['TIME','DATE']]


# Exclude any sentence that does not have the correct NER type of the answer type
# Returns : list of sentences that DO have correct NER type of the answer type
def process_doc(single_doc, q_type, doc_num):
	# in order of type constants (0 : who, 1: where, 2: when)

	# http://nbviewer.jupyter.org/github/gmonce/nltk_parsing/blob/master/1.%20NLTK%20Syntax%20Trees.ipynb
	def filter(x):
		for tag in NER_TAG[q_type]:
			return x.label() == tag

	sentences = nltk.tokenize.sent_tokenize(single_doc)
	surviving_sentences = []
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		pos_tag = nltk.pos_tag(words)
		# this is in nltk tree
		# reference : http://www.nltk.org/howto/tree.html
		# http://nbviewer.jupyter.org/github/gmonce/nltk_parsing/blob/master/1.%20NLTK%20Syntax%20Trees.ipynb
		ner_tree = nltk.ne_chunk(pos_tag)
		# bool for whether this sentence has
		contains_tag = False
		for subtree in ner_tree.subtrees(filter = filter):
			contains_tag = True
		if (contains_tag):
			surviving_sentences.append((doc_num,sentence))
	#print surviving_sentences
	return surviving_sentences


def passage_retrieval(q_num, q_type):
	# get the path for the docs for current question
	current_path = d_path + '/' + str(q_num) + '/*'
	#current_path = d_path + '/' + str(q_num) + '/*'
	# list of (doc_num, sentence) surviving after processing
	retrieved_sentences = []
	# To call glob in human sorting order
	# From : http://stackoverflow.com/a/16090640
	def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
		return [int(text) if text.isdigit() else text.lower()
			for text in re.split(_nsre, s)]

	for name in sorted(glob.glob(current_path), key=natural_sort_key):
		with open(name) as f:
			# bool for being inside text
			text_bool = False
			single_doc = ''
			for line in f:
				line = line.rstrip()
				if(line == '</TEXT>'):
					text_bool = False
				if(text_bool):
					single_doc += line + ' '
				if(line == '<TEXT>'):
					text_bool = True

			temp = name.split('/')
			doc_num = temp[len(temp)-1]
			
			# list of sentences
			retrieved_sentences += process_doc(single_doc, q_type, doc_num)
	#print retrieved_sentences
	#print doc_retrived_from
	return retrieved_sentences




###############################################################################
###############################################################################
# ANSWER PROCESSING
'''
BASELINE IMPLEMENTATION
	Method 1: Just return whatever matching entity
		If there are multiple matching entities, it returns the first one.
IN :  retrieved sentences
OUT : list of answers in string
'''

# Might have been better to return ner from previous part, and not
# do ner again, but left it in case some manipulation is needed for part 2
def answer_processing(sentences, q_type):
	# http://nbviewer.jupyter.org/github/gmonce/nltk_parsing/blob/master/1.%20NLTK%20Syntax%20Trees.ipynb
	def filter(x):
		for tag in NER_TAG[q_type]:
			return x.label() == tag
	# in string
	answers = []
	for i in range(0, 5):
		doc_num = sentences[i][0]
		sentence = sentences[i][1]
		words = nltk.word_tokenize(sentence)
		pos_tag = nltk.pos_tag(words)
		ner_tree = nltk.ne_chunk(pos_tag)
		print ner_tree
		# the list of tuples((word, pos),ner) to be considered for this sentence
		matching_tuples = []
		for subtree in ner_tree.subtrees(filter = filter):
			 matching_tuples = subtree.pos()
			 break
		# t : ((word, pos), ner)
		answer = ''
		for t in matching_tuples:
			print t
			answer += t[0][0] + ' '
		# remove any possible trailing whitespaces
		answer = answer.rstrip()
		answers.append((doc_num,answer))
	print answers
	return answers


###############################################################################
###############################################################################
# IO

# Types : WHO / WHERE / WHEN
# OR : PERSON/ LOCATION/ TIME
# 0 , 1 , 2 

def get_questions():
	with open(q_path) as f:
		num = 0
		desc_bool = False
		desc = ''
		q_dict = {}
		for line in f:
			line = line.rstrip()
			if desc_bool is True:
				desc = line
				q_dict[num] = desc
				desc_bool = False
			if "<num>" in line:
				split_s = line.split()
				num = int(split_s[len(split_s)-1])
			if "<desc>" in line:
				desc_bool = True
		return q_dict

def process_question(num, desc):
	print num
	print desc
	desc_token = nltk.word_tokenize(desc)
	#desc_pos = nltk.pos_tag(desc_token)
	l = get_q_keywords(desc_token)
	q_type = get_q_type(desc_token)
	retrieved_sentences = passage_retrieval(num, q_type)
	answers = answer_processing(retrieved_sentences, q_type)

def process_questions(q_dict):
	for i in range(89, 90):
		desc = q_dict[i]
		process_question(i, desc)

'''def generate_answer(num_desc):
	q_num, desc = num_desc
	path = d_path + '/' + str(q_num)
	for name in glob.glob('path/*'):
		with open(name) as f:
			for line in f:
				line = line.rstrip()
	#print desc
'''


q_dict = get_questions()
process_questions(q_dict)

