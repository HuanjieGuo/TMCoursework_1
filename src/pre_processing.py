
# 1 remove all the '?' of each sentence
# 2 split every line into 2 lists : label and sentences
# 3 lowercase the first letter of every sentence
# 4 将sentences进行tokenization并去除没有用的符号与词（初步停用词）

'''
整个预处理文件的使用：
// conf.get('param', 'path_train')表示传入的txt文件
1. labels, sentences = sentenceProcessing(conf.get('param', 'path_train'))
2. sentences = lower_first_letter(sentences)
'''

from src.question_classifier import conf

'''
sentenceProcessing方法
输入：txt文件
输出：两个list，labels, sentences
作用：将文件的每一行划分成独立的标签和句子
使用格式：labels, sentences = sentenceProcessing(conf.get('param', 'path_train'))
'''
def sentence_processing(file):
    labels = []
    sentences = []
    with open(file, 'r') as f:
        # result = f.read()
        # result = re.sub('[?]','',result)
        for line in f.readlines():
            line = line.strip('\n')
            # print(line)
            # print(line.split(' ',1))
            labels.append(line.split(' ', 1)[0])
            sentences.append(line.split(' ', 1)[1])
    return labels,sentences

'''
lower_first_letter方法：
输入：list，包含了多个句子
输出：小写化句子首字母的list
作用：小写化句子首字母
'''
def lower_first_letter(sentences):
    for i in range(len(sentences)):
        tmp = list(sentences[i])
        tmp[0] = tmp[0].lower()
        sentences[i] = ''.join(tmp)
    return sentences

'''
预处理：
'''
if __name__ == '__main__':
    labels,sentences =sentence_processing(conf.get('param', 'path_train'))
    sentences = lower_first_letter(sentences)

    print(len(labels))
    print(len(sentences))
