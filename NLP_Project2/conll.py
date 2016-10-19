
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
            # tag = 'B-CUE' if token[2] == 'I-CUE' else token[2]
            # word_token = token[0].lower().decode('utf-8') if token[0] not in single_occurance_word else '<UNK>'.decode('utf-8')
            sentence.append((token[0].lower().decode('utf-8'), token[1].decode('utf-8'), token[2]))
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
            tag = 'B-CUE' if token[2] == 'I-CUE' else token[2]
            sentence.append((token[0].lower().decode('utf-8'), tag))
            pos_tag.append((token[1].decode('utf-8'), tag)) 
        # throw away files that don't have 'B-CUE' or 'I-CUE'
        if 'B-CUE' in selector_flag or 'I-CUE' in selector_flag:
            formatted_sentence.append(sentence)
            formatted_pos_tag.append(pos_tag)
        if 'B-CUE' not in selector_flag and 'I-CUE' not in selector_flag and random.random() >= 1.1:
            formatted_sentence.append(sentence)
            formatted_pos_tag.append(pos_tag)
    # print len(formatted_sentence), len(formatted_pos_tag)
    return formatted_sentence, formatted_pos_tag

# process training set by BIO tagging and concatenate tokens into sentence
print '\n\n\nformatting training data set in progress...'
training_sentence, training_pos_tag = data_formater(training_set)

# process development set by BIO tagging and concatenate tokens into sentence
print '\n\n\nformatting development data set in progress...'
development_sentence, development_pos_tag = data_formater(development_set)

print(sklearn.__version__)


# # Let's use CoNLL 2002 data to build a NER system
# 
# CoNLL2002 corpus is available in NLTK. We use Spanish data.

# In[2]:

# nltk.corpus.conll2002.fileids()


# In[3]:

train_sents = training_sentence
test_sents = development_sentence


# train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


# Data format:

# In[4]:

# print train_sents[0]


# ## Features
# 
# Next, define some features. In this example we use word identity, word suffix, word shape and word POS tag; also, some information from nearby words is used. 
# 
# This makes a simple baseline, but you certainly can add and remove some features to get (much?) better results - experiment with it.

# In[5]:

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]    


# This is what word2features extracts:

# In[6]:

# print sent2features(train_sents[0])[0]


# Extract the features from the data:

# In[7]:

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

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


# Possible parameters for the default training algorithm:

# In[10]:

trainer.params()


# Train the model:

# In[11]:

trainer.train('conll2002-esp.crfsuite')


# trainer.train saves model to a file:

# In[12]:

# get_ipython().system(u'ls -lh ./conll2002-esp.crfsuite')


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


# Let's tag a sentence to see how it works:

# In[14]:

example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)))

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))


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
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


# Predict entity labels for all sentences in our testing set ('testb' Spanish data):

# In[16]:

y_pred = [tagger.tag(xseq) for xseq in X_test]


# ..and check the result. Note this report is not comparable to results in CONLL2002 papers because here we check per-token results (not per-entity). Per-entity numbers will be worse.  

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


# We can see that, for example, it is very likely that the beginning of an organization name (B-ORG) will be followed by a token inside organization name (I-ORG), but transitions to I-ORG from tokens with other labels are penalized. Also note I-PER -> B-LOC transition: a positive weight means that model thinks that a person name is often followed by a location.
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


# Some observations:
# 
# * **8.743642 B-ORG  word.lower=psoe-progresistas** - the model remembered names of some entities - maybe it is overfit, or maybe our features are not adequate, or maybe remembering is indeed helpful;
# * **5.195429 I-LOC  -1:word.lower=calle**: "calle" is a street in Spanish; model learns that if a previous word was "calle" then the token is likely a part of location;
# * **-3.529449 O      word.isupper=True**, ** -2.913103 O      word.istitle=True **: UPPERCASED or TitleCased words are likely entities of some kind;
# * **-2.585756 O      postag=NP** - proper nouns (NP is a proper noun in the Spanish tagset) are often entities.

# ## What to do next
# 
# 1. Load 'testa' Spanish data.
# 2. Use it to develop better features and to find best model parameters.
# 3. Apply the model to 'testb' data again.
# 
# The model in this notebook is just a starting point; you certainly can do better!
