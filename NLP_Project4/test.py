import similarity

word_list = similarity.uncertain_word()
for a in ['maybe', 'suggest', 'could', 'good', 'should', 'guess', 'haha', 'think']:
	print a, ' ', similarity.calculation(a, word_list)