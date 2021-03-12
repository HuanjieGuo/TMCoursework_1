### tokenization.py

```python
'''
To read the stop words from file.
 
return:
stop_list: a list of stop words.
'''
def read_stoplist()

'''
Split the sentences into tokens.

input:  
sentences: sentence list 
stop_list: a array of many stop words

return:
tokens: token set
token_of_sentences: each sentence's token
'''
def tokenization(sentences, stop_list)

'''
Delete ' ' and '``' in the input data

input:
oldlist: a list that contain ' ' and '``'

return:
newList: a list without ' ' and '``'
'''
def dellist(oldlist)
```



### sentence_vector.py

```python
'''
Use bag of word to sum all the vector of the tokens in one sentence and devide the count of token.

input:
tokens : a token list of a sentence
wordToIdx: a map that the key is token, and value is the idx of token in wordVec
wordVec: vectors matrix [n,vector_dimension] 

return :
vector_of_sentence: a vector that can represent the sentence.
'''
def make_bow_vector(tokens,wordToIdx,wordVec)

'''
Use a for iteration to call the 'make_bow_vector' function and append its result to a list

input:
sentences: a list of sentence.
wordToIdx: a map that the key is token, and value is the idx of token in wordVec.
wordVec: vectors matrix [n,vector_dimension].

return:
sentences_vector_list: a list of sentences' vector
'''
def multi_sentences_to_vectors(sentences,wordToIdx,wordVec)

'''
choose what kind of vector you want to get and it will return you train, dev and test data

input:
type: randomly or pre_train
freeze: True or False

return:
train_sentence_vectors
train_labels
dev_sentence_vectors
dev_labels
test_sentence_vectors
test_labels
'''
def bag_of_word_sentences(type='randomly',freeze=True)

'''
transform label form string to corresponding index

input:
train_labels
dev_labels
test_labels

return:
train_labels_idxs : label index
dev_labels_idxs
test_labels_idxs
'''
def get_label_number_to_idx(train_labels, dev_labels, test_labels)
```

### split_file.py

```python
'''
Split the 'train_5500.txt' into train.txt and dev.txt with a ratio of 9:1
'''
def get_train_dev()

'''
split data into two sets with a ratio

input:
full_list: total data
shuffle: do you wish to shuffle the data before split
ratio: 0-1, the ratio of the first set.

return:
sublist1
sublist2
'''
def random_split(full_list, shuffle=True, ratio=0)
```

### word_embedding.py

```python
'''
randomly generate all vector of the tokens

input:
tokens: token list
threshold: the token whose count is below threshold will be delete.

return: 
word_vectors: the vectors of words
wordToIx: a map that the key is the word, and value is its corresponding index.
'''
def randomly_initialised_vectors(tokens=None,threshold=None)

'''
To generate vector of tokens from pre_train model.

return: 
word_vectors: the vectors of words
wordToIx: a map that the key is the word, and value is its corresponding index.
'''
def get_pre_train_vector()
```

