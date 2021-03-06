from src.pre_processing import sentence_processing
from src.pre_processing import lower_first_letter
from src.tokenization import read_stoplist
from src.tokenization import tokenization
from src.question_classifier import conf
from src.word_embeddings import randomly_initialised_vectors
import numpy as np
def bag_of_words(wordToIdx,token_of_sentences,wordVec):
    vecOfSentences = []
    for sentences in token_of_sentences:
        vector = np.zeros(len(wordVec[0]))
        for token in sentences:
            if wordToIdx.__contains__(token):
                vector += wordVec[wordToIdx.get(token)]
        vecOfSentences.append(vector)
    vecOfSentences = np.array(vecOfSentences)
    print(vecOfSentences)


if __name__ == '__main__':
    labels, sentences = sentence_processing(conf.get('param', 'path_train'))
    sentences = lower_first_letter(sentences)

    read_stoplist = read_stoplist()
    tokens,token_of_sentences = tokenization(sentences,read_stoplist)

    #print('tokens: ', tokens)
    # print('token_of_sentences: ', len(token_of_sentences)) # 4905行

    # need
    wordVec,wordToIdx = randomly_initialised_vectors(tokens,threshold=20)
    # print(len(wordVec)) #  183 * 5

    bag_of_words(wordToIdx,token_of_sentences,wordVec)