# remove all the '?' of each sentence
# split every line into 2 lists : label and sentences
# lowercase the first letter of every sentence

from src.question_classifier import conf
# [param: file] input file (txt)
# 初步划分：每行 -> label + sentences
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

# 首字母小写
def lower_first_letter(sentences):
    for i in range(len(sentences)):
        tmp = list(sentences[i])
        tmp[0] = tmp[0].lower()
        sentences[i] = ''.join(tmp)
    return sentences

if __name__ == '__main__':
    labels,sentences =sentence_processing(conf.get('param', 'path_train'))
    sentences = lower_first_letter(sentences)

    print(len(labels))
    print(len(sentences))
