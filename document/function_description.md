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

### Bilstm_test.py

```python
'''
Generate the BiLSTM model and complete the training and testing, and save the model.
'''
def acc_(preds, y)

'''
Calculate the accuracy of a batch of sentences.

input:  
preds: Output from the BiLSTM model.
y: Real labels.

return:
acc: Average accuracy
'''
def tokenization(sentences, stop_list)

'''
def acc_(preds, y)

'''
Calculate the accuracy of a batch of sentences.

input:  
preds: Output from the BiLSTM model.
y: Real labels.

return:
acc: Average accuracy
'''
'''
def train_another_new(rnn, train_loader, optimizer, criteon)

'''
Training BiLSTM model.

input:  
rnn: BiLSTM model.
train_loader: Dataloader for training data set.
optimizer: Optimizer, Adam in the project.
criteon: Loss function, in this project is the cross entropy function.

return:
cells: Sentence representation of the current batch
'''
'''
def eval(rnn, test_loader, criteon, patience=50)

'''
Evaluate the accuracy of the model.

input:  
rnn: BiLSTM model.
test_loader: Dataloader for test data set.
criteon: Loss function, in this project is the cross entropy function.

return:
Print the average accuracy for the whole test data set.
'''
```

### bilistm_nn.py and bow_nn.py

```python
'''
Bilstm Classifier, BOW Classifier and four functions: for reading file, train the model, test the model and output the predict sentence to specified file
'''

class QuestionClassifier(nn.Module):
'''
Bilstm Classifier 
input layer: word_embedding_dim
output layer: count(unique label)
'''

class QuestionClassifier(nn.Module):
'''
BOW Classifier
input layer: word_embedding_dim
output layer: count(unique label)
'''
  
def readFile(file):
'''
function for reading files.

input:  
The path of the file to be read

return:
sentence vectors and labels
'''

def train():
'''
train the model for Bilstm classifier and save the model to specified path("path_model")

input:  
None

return:
None
'''

def test():
'''
test the model for Bilstm classifier at specified path("path_model") and output the accuracy.

input:  
None

return:
None
'''

def output_predict_sentence_to_file(label_list,correct_rate):
'''
out put the predict sentence to specified file("path_eval_result")

input:  
label_list: original sentences and their labels
correct_rate: correctness of the entire test set

return:
None
'''
```

