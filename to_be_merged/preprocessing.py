# import globals
from collections import Counter


def labels_extraction(data):
    labels = []
    sentences = []
    for line in data:
        s = line.split()
        labels.append(s[0])
        sentences.append(s[1:])
    return labels, sentences


def remove_punctuations(sentences):
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            if sentence[i] in "?''``,.&":
                sentence.remove(sentence[i])
            else:
                i += 1
    return sentences


def to_lower_case(sentences):
    for sentence in sentences:
        sentence[0] = sentence[0].lower()
    return sentences


def remove_stop_words(sentences):
    stop_words = []
    with open('stop_words.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            stop_words.append(line.strip('\n'))
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            if sentence[i] in stop_words:
                sentence.remove(sentence[i])
            else:
                i += 1
    return sentences


def make_vocabulary(sentences):
    all_words = []
    for sentence in sentences:
        all_words.extend(sentence)
    word_frequency = Counter(all_words)
    sorted_words = sorted(word_frequency, key=word_frequency.get, reverse=True)
    vocabulary = sorted_words
    return vocabulary


def get_preprocessed_sentences(file_path):
    with open(file_path, 'r',encoding="utf-8") as f:
        data = f.readlines()
        _, sentences = labels_extraction(data)
        sentences = remove_punctuations(sentences)
        return to_lower_case(sentences)


# def get_sorted_words():
#     return make_vocabulary(get_preprocessed_sentences())


if __name__ == '__main__':
    with open('train_1000.txt', 'r') as f:
        data = f.readlines()
        labels, sentences = labels_extraction(data)
        sentences = remove_punctuations(sentences)
        sentences = to_lower_case(sentences)
        # sentences = remove_stop_words(sentences)
        vocabulary = make_vocabulary(sentences)
        with open('vocabulary.txt', 'w') as fv:
            for word in vocabulary[:-1]:
                fv.write(word + '\n')
            fv.write(vocabulary[-1])
