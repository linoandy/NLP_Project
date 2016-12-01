
# coding: utf-8

# In[1]:

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import glob
import re
import nltk
import csv
import random
# import similarity
import json


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
            sentence.append((token[0].decode('utf-8'), token[1].decode('utf-8'), token[2]))
            pos_tag.append((token[1].decode('utf-8'), token[2]))
        formatted_sentence.append(sentence)
        formatted_pos_tag.append(pos_tag)
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
            # tag = 'B-CUE' if token[2] == 'I-CUE' else token[2]
            sentence.append((token[0].decode('utf-8'), token[1].decode('utf-8'), token[2]))
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

# this function get rid of all sentences that don't have B/I cues; however it doesn't work yet
def data_selector_formater_1(dataset):
    formatted_sentence = []
    formatted_pos_tag = []
    test_sentence_set = []
    test_pos_set = []
    for filename in dataset:
        with open(filename, 'r') as f:
            test_sentence = []
            test_pos = []
            prev_line_blank = False
            for line in f:
                words = line.split()

                if len(words) == 0 or words[0].decode('utf-8') == '.'.decode('utf-8'):
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
                    test_sentence.append([words[0].decode('utf-8'), words[1].decode('utf-8'), words[2]])
                    test_pos.append(words[1].lower().decode('utf-8'))

    def BIO(token_lists): # this function processes the document passed in, and replace CUE tags with BIO tags
        # replace '_\n' with tag 'O'
        for token_list in token_lists:
            if token_list[2] == '_':
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

    for sentence in test_sentence_set:
        sentence = BIO(sentence)

    for sentences in test_sentence_set:
        BI_exist = False
        for sentence in sentences:
            if sentence[2] == 'B-CUE' or sentence[2] == 'I-CUE':
                BI_exist = True
        if BI_exist == False:
            test_sentence_set.remove(sentences)
        print 'len(test_sentence_set)', len(test_sentence_set)

    return test_sentence_set, formatted_pos_tag

# process training set by BIO tagging and concatenate tokens into sentence
print '\n\n\nformatting training data set in progress...'
training_sentence, training_pos_tag = data_selector_formater(training_set)

# process development set by BIO tagging and concatenate tokens into sentence
print '\n\n\nformatting development data set in progress...'
development_sentence, development_pos_tag = data_formater(development_set)

train_sents = training_sentence
test_sents = development_sentence

# ## Features
# 
# Next, define some features. we use word identity, word shape and word POS tag; also, same information from nearby words is used. 
# In[5]:

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = []


    if i > 2:
        word1 = sent[i-3][0]
        postag1 = sent[i-3][1]
        features.extend([
            '-3:word.lower=' + word1.lower(),
            '-3:word.istitle=%s' % word1.istitle(),
            # '-3:word.isupper=%s' % word1.isupper(),
            '-3:postag=' + postag1,
            # '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][1]
        features.extend([
            '-2:word.lower=' + word1.lower(),
            '-2:word.istitle=%s' % word1.istitle(),
            # '-2:word.isupper=%s' % word1.isupper(),
            '-2:postag=' + postag1,
            # '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            # '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            # '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    uncertainty = word_list[word.encode('utf-8').lower()] if word.encode('utf-8').lower() in word_list else False
    features.extend([
        'word.lower=' + word.lower(),
        # 'word[-3:]=' + word[-3:],
        # 'word[-2:]=' + word[-2:],
        # 'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        # 'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'cue=%s' % uncertainty,
        # 'word.maymight=%s' % str(word.lower() in ['may', 'might', 'should', 'suggest', 'predict', 'likely', 'claim', 'consistently']),
        # 'postag[:2]=' + postag[:2],
    ])

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            # '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            # '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    if i < len(sent)-2:
        word1 = sent[i+2][0]
        postag1 = sent[i+2][1]
        features.extend([
            '+2:word.lower=' + word1.lower(),
            '+2:word.istitle=%s' % word1.istitle(),
            # '+2:word.isupper=%s' % word1.isupper(),
            '+2:postag=' + postag1,
            # '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    if i < len(sent)-3:
        word1 = sent[i+3][0]
        postag1 = sent[i+3][1]
        features.extend([
            '+3:word.lower=' + word1.lower(),
            '+3:word.istitle=%s' % word1.istitle(),
            # '+3:word.isupper=%s' % word1.isupper(),
            '+3:postag=' + postag1,
            # '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    # print 'features', features            
    return features


def sent2features(sent):
    # print 'sent', sent
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]    


# Extract the features from the data:

# In[7]:
with open('/Users/linoandy/GitHub/NLP_Project1/NLP_Project4/uncertainty.json') as json_data:
    word_list = json.load(json_data)
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# ## Train the model
# 
# To train the model, we create pycrfsuite.Trainer, load the training data and call 'train' method. 
# First, create pycrfsuite.Trainer and load the training data to CRFsuite:

# In[8]:

trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)


# Set training parameters. We will use L-BFGS training algorithm (it is default) with Elastic Net (L1 + L2) regularization.

# In[9]:

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier
    'feature.minfreq': 1, # ignore features that only occurred once

    # include transitions that are possible, but not observed
    # 'feature.possible_transitions': True
})


# Possible parameters for the default training algorithm:

# In[10]:

trainer.params()


# Train the model:

# In[11]:

trainer.train('conll2002-esp.crfsuite')


# trainer.train saves model to a file:

# In[12]:


# We can also get information about the final state of the model by looking at the trainer's logparser. If we had tagged our input data using the optional group argument in add, and had used the optional holdout argument during train, there would be information about the trainer's performance on the holdout set as well. 

# In[13]:

trainer.logparser.last_iteration


# We can also get this information for every step using trainer.logparser.iterations

# In[15]:

print len(trainer.logparser.iterations), trainer.logparser.iterations[-1]


# ## Make predictions
# 
# To use the trained model, create pycrfsuite.Tagger, open the model and use "tag" method:

# In[13]:

tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')


# Let's tag sentences in evaluation set to see and display the wrong predictions

# In[14]:
total_error_count = 0
totol_O_count = 0
for i in range(len(test_sents)):
    example_sent = test_sents[i]
    for j in range(len(example_sent)):
        if tagger.tag(sent2features(example_sent))[j] != sent2labels(example_sent)[j]:
            total_error_count += 1
            if tagger.tag(sent2features(example_sent))[j] == 'O':
                totol_O_count += 1
            print '(', example_sent[j][0], ', ', tagger.tag(sent2features(example_sent))[j], ', ', sent2labels(example_sent)[j],')'

print 'total_error_count', total_error_count
print 'O percentage', totol_O_count/float(total_error_count)

test_public_path = './nlp_project2_uncertainty/test-public/*.txt'
test_private_path = './nlp_project2_uncertainty/test-private/*.txt'

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
                    # test_pos_set.append(test_pos)
                    test_sentence = []
                    test_pos = []
                    continue
                else:
                    prev_line_blank = False
                    test_sentence.append((words[0].decode('utf-8'), words[1].decode('utf-8')))
                    # test_pos.append(words[1].lower().decode('utf-8'))

    prediction_result = []
    for i in range(len(test_sentence_set)):
        prediction_word = tagger.tag(sent2features(test_sentence_set[i]))
        prediction_result_temp = []
        for j in range(len(prediction_word)):
            prediction_result_temp.append((test_sentence_set[i][j][0], prediction_word[j]))
        prediction_result.append(prediction_result_temp)
    
    for single_sentence in prediction_result:
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
    with open('word_result_haha.csv', 'wb') as f:
        a = csv.writer(f)
        a.writerow(['Type', 'Spans'])
        a.writerow(['CUE-public'] + w_pu)
        a.writerow(['CUE-private'] + w_pr)
    with open('sentence_result_haha.csv', 'wb') as f:
        a = csv.writer(f)
        a.writerow(['Type', 'Indices'])
        a.writerow(['SENTENCE-public'] + s_pu)
        a.writerow(['SENTENCE-private'] + s_pr)
    return

word_result_pu, sentence_result_pu = test_model(test_public_path, 'crf')
word_result_pr, sentence_result_pr = test_model(test_private_path, 'crf')
write_to_csv(word_result_pu, word_result_pr, sentence_result_pu, sentence_result_pr)
# ## Evaluate the model

# In[15]:

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


# Predict entity labels for all sentences in our testing set:

# In[16]:

y_pred = [tagger.tag(xseq) for xseq in X_test]

# ..and check the result. 

# In[17]:

print(bio_classification_report(y_test, y_pred))


# ## Let's check what classifier learned

# In[18]:

from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


# 
# Check the state features:

# In[19]:

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])

