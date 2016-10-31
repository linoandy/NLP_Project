import glob

'''
	TODO 1. Question processing (list of keywords to IR)
		method 1: leave out the question word
		method 2: get the noun phrases
	TODO 2. Question Classification
		Easy for our project, just use the first word in question
		Who, where, when. Can be expanded to more categories in Part 2
	TODO 3. Query Reformulation
		I was going to do this, but questionable if this is necessary..
		This is pretty minor. Might just leave it out.
			EX) When was laser invented
				=> Laser was invented in..
	TODO 4. Passage Retrieval
		Since document retrieval was done already, just need to retrieve passages
		Extract sections, paragraphs, sentences from doc from retrived doc
		TODO 4.1 DO NER on the retrived, and exclude any passages that don't include the answer type
		TODO 4.2 RANK the passages
			Leave it for Part 2?
			Num of named entities of right type 
			Question keywords <- Maybe only this?
			Longest exact sequence of question keywords..
			Rank of doc, etc.
	TODO 5. Answer Processing
		From the retrieved passage, retrieve answer
		Method 1 : Just return whatever matching entity
		Method 2 : Handwritten regex patterns? <- for part 2 maybe
			Pattern learning????
'''

starting_num = 89
ending_num = 320
current_num = starting_num


q_path = "./question.txt"
a_path = "./answer.txt"
d_path = "./doc_dev"

###############################################################################
###############################################################################
# QUESTION PROCESSING

# IN  : Question Description
# OUT : List of keywords,
#       just leaving out the question word and stop word list(METHOD 1)

# using a small stop word list; can use a bigger one later if wanted
stop_word = ['a'
	]

def get_q_keywords(desc):
	return






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
