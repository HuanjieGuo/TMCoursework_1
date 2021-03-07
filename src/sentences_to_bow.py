from src.pre_processing import sentence_processing
from src.pre_processing import lower_first_letter
from src import tokenization
from src.question_classifier import conf
from src.word_embeddings import randomly_initialised_vectors
from src.bag_of_words import bag_of_words
from sklearn.svm import SVC
import numpy as np
def sentence_to_bow_train(path):
    # train set
    labels, sentences = sentence_processing(path)
    sentences = lower_first_letter(sentences)

    read_stoplist = tokenization.read_stoplist()
    tokens, token_of_sentences = tokenization.tokenization(sentences, read_stoplist)

    # need
    wordVec, wordToIdx = randomly_initialised_vectors(tokens, threshold=20)
    # print(len(wordVec)) #  183 * 5

    bagOfWord = bag_of_words(wordToIdx, token_of_sentences, wordVec)
    return bagOfWord,labels,wordVec,wordToIdx

def sentence_to_bow_test(path,wordToIdx,wordVec):
    # train set
    labels, sentences = sentence_processing(path)
    sentences = lower_first_letter(sentences)

    read_stoplist = tokenization.read_stoplist()
    tokens, token_of_sentences = tokenization.tokenization(sentences, read_stoplist)

    # print(len(wordVec)) #  183 * 5

    bagOfWord = bag_of_words(wordToIdx, token_of_sentences, wordVec)
    return bagOfWord,labels


if __name__ == '__main__':
    vector_train,label_train,wordVec,wordToIdx = sentence_to_bow_train(conf.get('param', 'path_train'))
    vector_test,label_test = sentence_to_bow_test(conf.get('param', 'path_test'),wordToIdx,wordVec)

    clf = SVC(kernel="linear")
    clf.fit(vector_train, label_train)

    print(clf.score(vector_test, label_test))




