

from src.preprocessing import sentence_processing, lower_first_letter
import re
from src.preprocessing import conf


'''
To read the stop words from file.
 
return:
stop_list: a list of stop words.
'''
def read_stoplist():
    file = conf.get('param', 'stop_words')
    stop_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            stop_list.append(line)
    return stop_list

'''
This function is used to split the sentences into tokens.

input:  
sentences: sentence list 
stop_list: a array of many stop words

return:
tokens: token set
token_of_sentences: each sentence's token
'''
def tokenization(sentences, stop_list):
    # split using whitespace
    token_of_sentences = []
    for i in range(len(sentences)):
        sentences[i] = sentences[i].split(' ')
        token_of_sentences.append(list(filter(lambda x: x != '?', sentences[i])))

    # 2-d array -> 1-d array
    sentences = [b for a in sentences for b in a]
    # 进一步去除: , :  " ' ! ? 's a the an
    reg = '[\s+ \, \. \: \" \' ! \?]+'
    for i in range(0, len(sentences)):
        sentences[i] = re.sub(reg, '', sentences[i])
        # if sentences[i].lower() == 'a' or sentences[i].lower() == 'an' or sentences[i].lower() == 'the':
        #     sentences[i] = ''
        if (sentences[i].lower() in stop_list):
            sentences[i] = ''
    for sen in range(0, len(token_of_sentences)):
        for i in range(0, len(token_of_sentences[sen])):
            token_of_sentences[sen][i] = re.sub(reg, '', token_of_sentences[sen][i])
            if token_of_sentences[sen][i].lower() in stop_list:
                token_of_sentences[sen][i] = ''

    
    tokens = [i for i in sentences if i != '' and i != '``']
    token_of_sentences = dellist(token_of_sentences)
    return tokens, token_of_sentences

'''
Delete ' ' and '``' in the input data

input:
oldlist: a list that contain ' ' and '``'

return:
newList: a list without ' ' and '``'
'''
def dellist(oldlist):
    new2 = []
    for sen in oldlist:
        new1 = []
        for x in sen:
            if x != '' and x != '``':
                new1.append(x)
        new2.append(new1)
    return new2


if __name__ == '__main__':
    labels, sentences = sentence_processing(conf.get('param', 'path_train'))
    sentences = lower_first_letter(sentences,conf.get('param','lowercase'))
    stop_list = read_stoplist()
    tokens, token_of_sentences = tokenization(sentences, stop_list)

    print('tokens: ', tokens)
    print('token_of_sentences: ', token_of_sentences)
