import glob
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz
from threading import Thread
# import the following customized script to tag DATE NER
import timex
from spacy.en import English

parser = English()
'''
	PART 1. Question processing (list of keywords to IR)
		method 1: leave out the question word <- Method chosen for baseline
		method 2: get the noun phrases
		Baseline implementation : method 1 + removing stop words with nltk stopwords + removing punct
		Implemented this, but DIDN't it for baseline.
		Possible addition : stemming?
	PART 2. Question Classification
		Easy for our project, just use the first word in question
		Who, where, when. Can be expanded to more categories in Part 2
			Especially:
				Who was .. is asking for a description
				Who did .. //Who (other verbs) is asking for a person; currently only searching person
		When NER doesn't work now. nltk ner doesn't recognize time/date.
		Might need stanford ner...
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
	PART 5. Answer Processing
		From the retrieved passage, retrieve answer
		Method 1 : Just return whatever matching entity
		Method 2 : Handwritten regex patterns? <- for part 2 maybe
			Pattern learning????
		Baseline implementation : method 1. Used those early in the list first, so in the order
		of documents
'''


starting_num = 89
ending_num = 320
current_num = starting_num
window_keyword = 1

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

def syn_keywords(l_stopped):
	keywords = []
	for l in l_stopped:
		keywords.append(l);
		synsets = wn.synsets(l)
		for s in synsets:
			lems = map(lambda x : x.name().replace('_',' '), s.lemmas())
			keywords+=lems	
	return list(set(keywords))

def get_dependencies(desc):
	# TO RETURN : pass/active between main verb and subject
	'''
	a = grammar_parser.raw_parse(desc)
	for line in a:
		print line
	b = dependency_parser.raw_parse(desc)
	d_list =  [list(parse.triples()) for parse in b]
	print 'DLIST'
	main_dependency = d_list[0][0][1]'''
	'''
	parse_list =  [parse.tree() for parse in dependency_parser.raw_parse(desc)]
	tree = parse_list[0]
	main_verb = filter(lambda x : x.isalpha(), str(tree).split(' ')[0])
	print main_verb'''
	#https://github.com/NSchrading/intro-spacy-nlp/blob/master/Intro_spaCy_NLP.ipynb
	parsed = parser(unicode(desc))

	main_verb = ''

	for token in parsed:
		print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])

		if token.dep_ == 'nsubj':
			return 'nsubj'
		if token.dep_ == 'nsubjpass':
			return 'nsubjpass'

	return ''

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
	# print first_word
	if(first_word == 'Who'):
		return WHO_TYPE
	if(first_word == 'Where'):
		return WHERE_TYPE
	if(first_word == 'When'):
		return WHEN_TYPE
	return -1


def rank_sentences_complex(question_keywords, question_desc, retrieved_sentences):
	parsed_q = parser(unicode(question_desc))
	dep_q = ''
	for token in parsed_q:
		dep_q += token.dep_ + ' '
	#print 'DEPQ'
	#print dep_q
	sentences = []
	for sentence in retrieved_sentences:
		parsed_a = parser(unicode(sentence))
		a_doc_num = sentence[0]
		a_sentence = sentence[1]
		syntax_score = 0.0
		keyword_score = 0.0


		dep = ''
		for token in parsed_a:
			dep += token.dep_ + ' '
		#print dep
		syntax_score = fuzz.partial_ratio(dep_q, dep)
		for question_keyword in question_keywords:	
			similarity_score = fuzz.partial_ratio(question_keyword.lower(), a_sentence.lower())
			keyword_score += similarity_score
		#print syntax_score
		#print keyword_score

		score = syntax_score * 0.05  + keyword_score * .95


		sentences.append([a_doc_num, a_sentence, score])
	sorted_sentence_list_temp = sorted(sentences, key=lambda x: x[2], reverse=True)
	sorted_sentence_list = []
	for sorted_sentence_temp in sorted_sentence_list_temp:
		sorted_sentence = (sorted_sentence_temp[0], sorted_sentence_temp[1])
		sorted_sentence_list.append(sorted_sentence)
	return sorted_sentence_list




def rank_sentences_grammar(question_desc, answer_sentences):
	parsed_q = parser(unicode(question_desc))
	dep_q = ''
	for token in parsed_q:
		dep_q += token.dep_ + ' '
	#print 'DEPQ'
	#print dep_q
	sentences = []
	for sentence in answer_sentences:
		parsed_a = parser(unicode(sentence))
		a_doc_num = sentence[0]
		a_sentence = sentence[1]
		score = 0.0

		dep = ''
		for token in parsed_a:
			dep += token.dep_ + ' '
		#print 'DEP'
		#print dep
		score = fuzz.partial_ratio(dep_q, dep)
		#print 'SCORE'
		#print score
		sentences.append([a_doc_num, a_sentence, score])
	sorted_sentence_list_temp = sorted(sentences, key=lambda x: x[2], reverse=True)
	sorted_sentence_list = []
	for sorted_sentence_temp in sorted_sentence_list_temp:
		sorted_sentence = (sorted_sentence_temp[0], sorted_sentence_temp[1])
		sorted_sentence_list.append(sorted_sentence)
	return sorted_sentence_list



def rank_sentences(question_keywords, answer_sentences):
	# question_keywords ['xxxx', 'xxxxxx']
	# answer_sentences [(doc_num, sentence), (doc_num, sentence)]
	keyword_sentences = []
	for answer_sentence in answer_sentences:
		a_doc_num = answer_sentence[0]
		a_sentence = answer_sentence[1]
		total_score = 0.0
		# calculate similarity scores between different keywords and sentences respectively, sum up the scores as the final ranking 
		for question_keyword in question_keywords:	
			similarity_score = fuzz.partial_ratio(question_keyword.lower(), a_sentence.lower())
			total_score += similarity_score
		keyword_sentences.append([a_doc_num, a_sentence, total_score])
	# sort the result sentence in descending order by sum scores 
	sorted_sentence_list_temp = sorted(keyword_sentences, key=lambda x: x[2], reverse=True)
	# build the result into its old format, same as answer_sentences
	sorted_sentence_list = []
	for sorted_sentence_temp in sorted_sentence_list_temp:
		sorted_sentence = (sorted_sentence_temp[0], sorted_sentence_temp[1])
		sorted_sentence_list.append(sorted_sentence)
	return sorted_sentence_list

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
	#print "DOING PROCESS_DOC"
	# in order of type constants (0 : who, 1: where, 2: when)

	# BASELINE ONLY!
	# when NER doesn't work in nltk NER. we use timex.py to tag
	# https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/timex.py
	if q_type == WHEN_TYPE:
		sentences = nltk.tokenize.sent_tokenize(single_doc)
		surviving_sentences = []
		for sentence in sentences:
			sentence_after_tagging = timex.tag(sentence)
			if sentence_after_tagging.find('<TIMEX2>') != -1:
				surviving_sentences.append((doc_num,sentence))
		return surviving_sentences

	# http://nbviewer.jupyter.org/github/gmonce/nltk_parsing/blob/master/1.%20NLTK%20Syntax%20Trees.ipynb

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
		for subtree in ner_tree.subtrees():
			if subtree.label() in NER_TAG[q_type]:
				contains_tag=True
		if (contains_tag):
			surviving_sentences.append((doc_num,sentence))
	#print surviving_sentences
	return surviving_sentences


def passage_retrieval(q_num, q_type, q_keywords):
	#print "DOING PASSAGE_RETRIEVAL"
	# get the path for the docs for current question
	current_path = d_path + '/' + str(q_num) + '/*'
	wnl = WordNetLemmatizer()
	#current_path = d_path + '/' + str(q_num) + '/*'
	# list of (doc_num, sentence) surviving after processing
	retrieved_sentences = []
	# To call glob in human sorting order
	# From : http://stackoverflow.com/a/16090640
	def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
		return [int(text) if text.isdigit() else text.lower()
			for text in re.split(_nsre, s)]
	# print "KEYWORDS", q_keywords		
	# print q_keywords, "KEYWORD FIRST"
	q_keywords = map(lambda x : wnl.lemmatize(x.lower()),q_keywords)		
	for name in sorted(glob.glob(current_path), key=natural_sort_key):
		# global doc
		with open(name) as f:
			doc = f.read().decode("ascii","ignore").encode("ascii").rstrip()
		single_doc=''
		# Fetch TEXT only. a bit hack tho - need a better way to process text
		# Did this because I wanted to fetch text by 'sentence' not 'line'
		# - Rommie, 161104
		doc = doc.replace('\r\n',' ')
		
		sentlist = nltk.tokenize.sent_tokenize(doc)
			# print name
			# bool for being inside text
			# text_bool = False
			# single_doc = ''
		i=0
		
		while i < len(sentlist):
			line = sentlist[i]
			tmpline = line.lower().strip()
			tmpline = nltk.tokenize.word_tokenize(tmpline)
			tmpline = map(lambda x : wnl.lemmatize(x), tmpline)
			iskwin = map(lambda x : x in tmpline, q_keywords)			
			if any(iskwin):
				k=0
				# print line
				while (k<window_keyword) and i+k < len(sentlist):
					# print sentlist[i+k]
					single_doc += sentlist[i+k] + ' '
					k+=1
				i=i+k-1
			i+=1

		# for line in sentlist:
		# 	tmpline = line.lower().strip().decode("ascii","ignore").encode("ascii").rstrip()
		# 	tmpline = nltk.tokenize.word_tokenize(tmpline)
		# 	tmpline = map(lambda x : wnl.lemmatize(x), tmpline)
		# 	# print tmpline, "TMPLINE"
		# 	# print q_keywords, "KEYWORD"
		# 	#is keyword in a sentence?
		# 	iskwin = map(lambda x : x in tmpline, q_keywords)
		# 	if any(iskwin):
		# 		single_doc += line + ' '
		temp = name.split('/')
		doc_num = temp[len(temp)-1]
		#print single_doc
		# list of (doc_num, sentences)
		retrieved_sentences += process_doc(single_doc, q_type, doc_num)
		# IN ORDER TO SAVE TIME FOR BASELINE
		# NEED TO REMOVE THIS FOR MORE COMPLICATED SHIT
		# possible because baseline only takes the retrieved sentecnes in order
		if len(retrieved_sentences) >= 30:
			break
	return retrieved_sentences


###############################################################################
###############################################################################
# ANSWER PROCESSING
'''
BASELINE IMPLEMENTATION
	Method 1: Just return whatever matching entity
		If there are multiple matching entities, it returns the first one.
IN :  retrieved sentences tuple (doc_num, sentence)
OUT : list of (doc_num, answer), both in string
'''

def grammar_stuff(answer_list, answer_sentence, dependency, doc_num):
	passed_answers = []
	failed_answers = []
	parsed = parser(unicode(answer_sentence))
	for answer in answer_list:
		#print 'answer'
		#print answer
		found = False
		answer_words = answer.split()
		for answer_word in answer_words:
			#print 'answer word'
			#print answer_word
			if found:
				break
			for token in parsed:
				#print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
				if token.orth_ == answer_word or token.head.orth_ == answer_word:
					if token.dep_ == dependency:
						passed_answers.append((doc_num,answer))
						found = True
						break
		if not found:
			failed_answers.append((doc_num,answer))


	return passed_answers, failed_answers

	'''
	if ('<' in answer_sentence):
		raise Exception('tags messing with grammar')

	parse_list =  [parse.tree() for parse in dependency_parser.raw_parse(answer_sentence)]
	tree = parse_list[0]
	main_verb = filter(lambda x : x.isalpha(), str(tree).split(' ')[0])
	parse_list = [parse.tree() for parse in dependency_parser.raw_parse(answer_sentence)]
	main_verb = filter(lambda x : x.isalpha(), str(parse_list[0]).split(' ')[0])
	print main_verb'''
	'''
	d_list =  [list(parse.triples()) for parse in dependency_parser.raw_parse(answer_sentence)]
	print d_list
	for answer in answer_list:
		print 'DEPPPP'
		print dependency
		print 'ANSSSWER'
		print answer
		#print d_list[0][0]
		answer_words = answer.split()
		bool_break = False

		for answer_word in answer_words:
			if bool_break:
				break
			for i in range(0, len(d_list[0])):
				triplet = d_list[0][i]
				#print 'triplet!!!!'
				#print triplet
				w1 = triplet[0][0]
				dep = triplet[1]
				w2 = triplet[2][0]
				
				if((w1 == answer_word and w2 == main_verb) or (w2 == answer_word and w1 == main_verb)):
					if(dep == dependency):
						passed_answers.append((doc_num,answer))
						bool_break = True
						break
		if bool_break == False:
			failed_answers.append((doc_num,answer))
	print 'PASSED!'
	print passed_answers
	'''

# Might have been better to return ner from previous part, and not
# do ner again, but left it in case some manipulation is needed for part 2
def answer_processing(s_tuple, q_type, q_keywords, dependency):
	#print "DOING ANSWER_PROCESSING"
	sentences = s_tuple
	print len(sentences)
	# http://nbviewer.jupyter.org/github/gmonce/nltk_parsing/blob/master/1.%20NLTK%20Syntax%20Trees.ipynb
	# in string
	answers = []

	grammar_passed_answers = []
	grammar_failed_answers = []


	# NEED TO ACCOUNT FOR CASES IN WHICH THERE ARE LESS THAN 5 ANSWERS
	num_answers_needed = 5 - len(sentences)
	if(num_answers_needed > 0):
		for i in range(0,num_answers_needed):
			sentences.append(('100','nil'))
	for i in range(0, min(10, len(sentences))):
		doc_num = sentences[i][0]
		sentence = sentences[i][1]
		if q_type == WHEN_TYPE:
			sentence_after_tagging = timex.tag(sentence)
			when_answers = re.findall('<TIMEX2>(.*?)</TIMEX2>', sentence_after_tagging)
			# in case answer comes out as empty, output an empty string
			when_answer = when_answers[0] if len(when_answers) != 0 else 'nil'
			answers.append((doc_num, when_answer))

		else:		
			words = nltk.word_tokenize(sentence)
			pos_tag = nltk.pos_tag(words)
			ner_tree = nltk.ne_chunk(pos_tag)
			#print ner_tree
			# the list of tuples((word, pos),ner) to be considered for this sentence
			matching_tuples = []
			# print q_keywords
			global subtree
			tmp = []
			for subtree in ner_tree.subtrees():
				if subtree.label() in NER_TAG[q_type] and subtree.pos()[0][0][1]=='NNP':					
					word = ' '.join(map(lambda x : x[0][0], subtree.pos()))
					#print word
					#print q_keywords
					iskwin = map(lambda x : x in word, q_keywords)
					if not any(iskwin):						
						# print "SUBTREE!", subtree
						# matching_tuples = subtree.pos()
						answer = ' '.join(map(lambda x : x[0][0], subtree.pos()))
						if answer not in map(lambda x : x[1],answers):
							tmp.append(answer)
			
			if(len(tmp) > 0 and dependency != '' and q_type == WHO_TYPE):
				try:
					p, f = grammar_stuff(tmp,sentence, dependency, doc_num)
					grammar_passed_answers += p
					grammar_failed_answers += f
				except:
					for answer in tmp:
						grammar_failed_answers.append((doc_num, answer))
			else:
				for answer in tmp:
					grammar_failed_answers.append((doc_num,answer))
			'''
			for answer in tmp:
				grammar_failed_answers.append((doc_num,answer))
			'''
					
			#print "SENTENCE : ", sentence, "ANSWER : ", tmp
			# t : ((word, pos), ner)
			# answer = ''
			# for t in matching_tuples:
			# 	#print t
			# 	if t[0][0] not in q_keywords:
			# 		answer += t[0][0] + ' '
			# # remove any possible trailing whitespaces
			# answer = answer.rstrip()
			# answers.append((doc_num,answer))

	answers += grammar_passed_answers + grammar_failed_answers
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
	# print num
	print desc
	desc_token = nltk.word_tokenize(desc)
	#desc_pos = nltk.pos_tag(desc_token)
	l = get_q_keywords(desc_token)
	q_type = get_q_type(desc_token)
	main_dependency = get_dependencies(desc)
	retrieved_sentences = passage_retrieval(num, q_type,l)
	# add sentence ranking below
	syn_kw = syn_keywords(l)
	#retrieved_sentences = rank_sentences(l,retrieved_sentences)
	#retrieved_sentences = rank_sentences_grammar(desc,retrieved_sentences)
	retrieved_sentences = rank_sentences_complex(l, desc, retrieved_sentences)

	answers = answer_processing(retrieved_sentences, q_type, l, main_dependency)
	# answers : (doc_num, answers) tuple; both string
	return answers

def answer_output(answer_tuple):
	with open(a_path, 'w') as f:
		for single_question in answer_tuple:
			q_num = single_question[0]
			#print q_num
			#print type(q_num)
			count = 0

			# now iterating (doc_num, answer)
			for answers in single_question[1]:
				count+=1
				if count >5:
					break
				doc_num = answers[0]
				answer = answers[1]
				#print doc_num
				#print answer
				f.write(q_num + ' ' + doc_num + ' ' + answer + '\n')
	return

def process_questions(q_dict):
	# [(q_num_0,[(doc_num_0, answer_0),.. (doc_num_4, answer_4)],(q_num_1,... ]
	q_num_answer_tuple = []
	for i in range(89, 321):
		desc = q_dict[i]
		answers = process_question(i, desc)
		q_num_answer_tuple.append((str(i), answers))
	# print q_num_answer_tuple
	answer_output(q_num_answer_tuple)

# def process_questions(q_dict):
# 	def main(j, q_dict):
# 		desc = q_dict[j]
# 		answers = process_question(j, desc)
# 		q_num_answer_tuple.append((str(j), answers))

# 	list_threads = []
# 	q_num_answer_tuple = []
# 	for i in range(89, 321):
# 		t = Thread(target=main, args=(i, q_dict))
# 		list_threads.append(t)
# 		t.start()
# 		if len(list_threads) >= 5:
# 			for z in list_threads:
# 				z.join()
# 			list_threads = []
# 	# print q_num_answer_tuple
# 	answer_output(q_num_answer_tuple)



q_dict = get_questions()
process_questions(q_dict)



