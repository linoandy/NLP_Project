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

# Types : WHO / WHERE / WHEN
# OR : PERSON/ LOCATION/ TIME
# 0 , 1 , 2 

def get_questions():
	with open(q_path) as f:
		num = 0
		desc_bool = False
		desc = ''
		num_desc = []
		for line in f:
			line = line.rstrip()
			if desc_bool is True:
				desc = line
				num_desc.append((num, desc))
				desc_bool = False
			if "<num>" in line:
				split_s = line.split()
				num = int(split_s[len(split_s)-1])
			if "<desc>" in line:
				desc_bool = True
		return num_desc

# get question type for the current question
# WHO / WHERE / WHEN
#def get_question_type():


'''def get_inverted(desc):
	# who / where / when
	#  0  /   1   /  2
	l = desc.split()
	q_type = -1
	inv_s = ''
	print l
	if l[0] == 'Who':
		q_type = 0
	if l[0] == 'Where':
		q_type = 1
	if l[0] is'When':
		q_type = 2
	print q_type

	# original : Who is Colin Powell?
	# modified : Colin Powell is
	if q_type == 0:
		for i in range(1, len(l)):
			inv_s += l[i] + ' '
		inv_s += l[1]

	if q_type == 1:
		# has located? at the end
		# original: Where is .. located?
		# modified: .. is located in
		if l[len(l)-1] == 'located?':
			for i in range(2, len(l)-1):
				inv_s += l[i] + ' '
			inv_s += l[1]
			inv_s += ' located in'
		# does not have located? at the end
		# original: Where is ..?
		# modified: 
		else:
			for i in range(2, len(l)):
				if i == len(l)-1:
					l[i] = l[i].replace('?', '')
				inv_s += l[i] + ' '
			inv_s += l[1]
			inv_s += ' located in'
	#TODO
	#if q_type == 2:
	return inv_s
'''

def generate_answer(num_desc):
	q_num, desc = num_desc
	path = d_path + '/' + str(q_num)
	#inv = get_inverted(desc)
	#print inv
	for name in glob.glob('path/*'):
		with open(name) as f:
			for line in f:
				line = line.rstrip()
	#print desc




num_desc = get_questions()

generate_answer(num_desc[0])