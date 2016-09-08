import re
import glob
from nltk import tokenize

corpus_sentence = []
corpus_word = []

corpusName = raw_input('Please enter the type of corpus: ')
path = "./data_corrected/classification_task/%s/train_docs/*.txt" % (corpusName)
for filename in glob.glob(path):
	with open(filename, 'r') as f:
		for line in f:
			# find "From : email" and replace them with empty string
			email = re.findall(r"From\s*:\s*[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\s", line)
			if len(email) > 0:
				line = line.replace(email[0].encode('utf-8'), '')

			# find first "Subject : " and replace with empty string
			line = line.replace('Subject : ', '', 1)

			# find ' >' and replace with empty string
			line = line.replace(' >', '')

			# build corpus of sentences
			corpus_sentence += tokenize.sent_tokenize(line)

			# build corpus of words
			corpus_word += line.split(' ')

print corpus_word # this is a list of words
print corpus_sentence # this is a list of sentences


