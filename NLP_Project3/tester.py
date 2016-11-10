from fuzzywuzzy import fuzz

def rank_sentences(question_keywords, answer_sentences):
	keyword_sentences = []
	for answer_sentence in answer_sentences:
		a_doc_num = answer_sentence[0]
		a_sentence = answer_sentence[1].lower()
		total_score = 0.0
		for question_keyword in question_keywords:	
			similarity_score = fuzz.partial_ratio(question_keyword, a_sentence)
			total_score += similarity_score
		keyword_sentences.append([a_doc_num, a_sentence, total_score])
	sorted_sentence_list_temp = sorted(keyword_sentences, key=lambda x: x[2], reverse=True)
	sorted_sentence_list = []
	for sorted_sentence_temp in sorted_sentence_list_temp:
		sorted_sentence = [sorted_sentence_temp[0], sorted_sentence_temp[1]]
		sorted_sentence_list.append(sorted_sentence)
	return sorted_sentence_list

test = ["translated", "airplane"]
sent = [('1', "and we translated that to what we thought was the most basic of all airplanes, the paper airplane,'' said Tim McClure, executive creative director of Austin advertising agency GSD&amp;M, which designed the report."), ('1', "The booklet, besides reporting on Southwest's record $57.95 million in profit, traces the construction of a paper airplane from the first folding of the paper to the final painting to resemble Shamu One, Southwest's ``flying whale'' jet rolled out last year during the airline's promotion of Sea World theme parks in Texas and California."), ('1', 'asked Southwest spokeswoman Charlotte Goddard.'), ('1', 'Besides a picture of Chairman Herb Kelleher throwing a paper airplane, the middle of the report features a four-page foldout picture of Shamu One.'), ('1', "On the back page, there's the punch-out version of the unusually painted plane, with assembly instructions and this ``helpful hint:''    ``Herb attaches a paper clip to the nose for longer flights (Shamu One's, not his).''"), ('1', "Even though Southwest previously has taken a light approach to its reports, McClure said it was a little difficult this year to sell the idea of a punch-out airplane because, ``once someone punches it out, you have an annual report with holes in the back.''"), ('1', 'He said he persuaded Southwest that the unusual idea would become enticing. '), ('3', 'The airplane was attempting to land at nearby Coronado Airport, Porter said.'), ('3', 'Walter Ramazzini Jr., 17, of Albuquerque, said he was sitting about 100 yards from the airplane when it crashed.'), ('3', "and went nose-first into the ground,'' said Ramazzini, a student pilot."), ('3', "The airplane ``wasn't very high _ maybe 100 feet,'' Ramazzini said.")]
print rank_sentences(test, sent)