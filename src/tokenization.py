'''
tokenization：
1. 将sentences进行分词
2. 并去除了一些停用词    , :  " ' ! ? 's a the an
输入：一个list，这个list由很多个句子组成 ['sentences1', 'sentences2', ...]
输出：list，由token组成
'''
'''
该文件的使用，基于preProcessing，流程：
1. 划分：labels, sentences = sentenceProcessing(conf.get('param', 'path_train'))
2. 小写化首字母：sentences = lower_first_letter(sentences)
3. Tokenization: tokens = tokenization(sentences)
'''

from src.pre_processing import sentence_processing, lower_first_letter
import re
from src.pre_processing import conf

def tokenization(sentences):
    # split using whitespace
    for i in range(len(sentences)):
        sentences[i] = sentences[i].split(' ')
    # 2-d array -> 1-d array
    sentences = [b for a in sentences for b in a]
    # 进一步去除: , :  " ' ! ? 's a the an
    for i in range(0, len(sentences)):
        reg = '[\s+ \, \: \" \' ! \?]+'
        sentences[i] = re.sub(reg, '', sentences[i])
        if sentences[i].lower() == 'a' or sentences[i].lower() == 'an' or sentences[i].lower() == 'the':
            sentences[i] = ''
    # 删除空元素''和'``'
    sentences = [i for i in sentences if i != '' and i != '``']
    return sentences


'''
Tokenization 使用流程：
'''
if __name__ == '__main__':
    labels, sentences = sentence_processing(conf.get('param', 'path_train'))
    sentences = lower_first_letter(sentences)
    tokens = tokenization(sentences)

    print('tokens: ', tokens)
