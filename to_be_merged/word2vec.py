# By Maxine

import preprocessing
from collections import Counter
import random

import numpy as np
import torch
from torch import nn
import torch.optim as optim


def create_dict(sorted_words):
    word_idx = {sorted_words[i]: i for i in range(len(sorted_words))}
    idx_word = {i: sorted_words[i] for i in range(len(sorted_words))}
    # word_idx['unknown_'] = len(sorted_words)
    # idx_word[len(sorted_words)] = 'unknown_'
    return word_idx, idx_word


def replace_words_with_idx(sentences, word_idx):
    sentences_in_idx = []
    for sentence in sentences:
        sentences_in_idx.append([word_idx[word] for word in sentence])
    return sentences_in_idx


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim, noise_dist=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.noise_dist = noise_dist

        self.in_embedding_layer = nn.Embedding(vocab_size, dim)
        self.out_embedding_layer = nn.Embedding(vocab_size, dim)

        self.in_embedding_layer.weight.data.uniform_(-1, 1)
        self.out_embedding_layer.weight.data.uniform_(-1, 1)

    def input_forward(self, input_words):
        return self.in_embedding_layer(input_words)

    def output_forward(self, output_words):
        return self.out_embedding_layer(output_words)

    def noise_forward(self, batch_size, n_samples):
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist

        # Sample words from our noise distribution
        noise_words = torch.multinomial(
            noise_dist, batch_size*n_samples, replacement=True).to('cpu')
        noise_vectors = self.out_embedding_layer(
            noise_words).view(batch_size, n_samples, self.dim)
        return noise_vectors


class NegativeSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, dim = input_vectors.shape
        input_vectors = input_vectors.view(batch_size, dim, 1)
        output_vectors = output_vectors.view(batch_size, 1, dim)

        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()

        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(),
                               input_vectors).sigmoid().log()
        # sum the losses over the sample of noise vectors
        noise_loss = noise_loss.squeeze().sum(1)

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()


def get_total_word_frequency(sentences):
    all_words = []
    for sentence in sentences:
        all_words.extend(sentence)
    total_count = len(all_words)
    word_count = Counter(all_words)
    word_frequency = {word: count/total_count for word,
                      count in word_count.items()}
    # word_frequency['unknown_'] = 0
    return word_frequency


def get_target_words(context, pos, max_window_size=5):
    window_size = np.random.randint(1, max_window_size)
    left = pos-window_size if pos-window_size >= 0 else 0
    right = pos+window_size+1 if pos + \
        window_size < len(context) else len(context)
    return context[left:pos]+context[pos+1:right]


def generate_batches(sentences, batch_size, max_window_size=5):
    corpus = []
    for sentence in sentences:
        corpus.extend(sentence)
    n_batches = len(corpus)//batch_size
    corpus = corpus[:n_batches*batch_size]

    for pos in range(0, len(corpus), batch_size):
        batch = corpus[pos:pos+batch_size]
        x, y = [], []
        for i in range(batch_size):
            input_word = batch[i]
            target_words = get_target_words(batch, i, max_window_size)
            x.extend([input_word]*len(target_words))
            y.extend(target_words)
        yield x, y


def cosine_similarity(embedding, valid_size=16, valid_window=100):
    # Here we're calculating the cosine similarity between some random words and
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.

    # sim = (a . b) / |a||b|

    embed_vectors = embedding.weight

    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent
    valid_examples = np.array(random.sample(
        range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(
        range(1000, 1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to('cpu')

    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes

    return valid_examples, similarities


def train(vocabulary_size, embedding_dim, sentences, idx_word, batch_size=200, epochs=10):
    word_frequency = get_total_word_frequency(sentences)
    sorted_frequencies = np.array(sorted(word_frequency.values(), reverse=True))

    # unigram distribution: according to the frequency that each word shows up in our corpus
    unigram_dist = sorted_frequencies/sorted_frequencies.sum()
    noise_dist = torch.from_numpy(unigram_dist**0.75/np.sum(unigram_dist**0.75))

    model = SkipGram(vocabulary_size, embedding_dim, noise_dist).to('cpu')
    negative_sampling_loss = NegativeSampling()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    steps = 0
    for e in range(epochs):
        for input_words, target_words in generate_batches(sentences, batch_size):
            steps += 1
            inputs, targets = torch.LongTensor(input_words).to('cpu'), torch.LongTensor(target_words).to('cpu')

            input_vectors = model.input_forward(inputs)
            output_vectors = model.output_forward(targets)
            noise_vectors = model.noise_forward(inputs.shape[0], 5)

            loss = negative_sampling_loss(input_vectors, output_vectors, noise_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % 100 == 0:
                print("Epoch: {}/{}".format(e+1, epochs))
                # avg batch loss at this point in training
                print("Loss: ", loss.item())
                valid_examples, valid_similarities = cosine_similarity(model.in_embedding_layer)
                _, closest_idxes = valid_similarities.topk(5)

                valid_examples, closest_idxes = valid_examples.to('cpu'), closest_idxes.to('cpu')
                for i, valid_idx in enumerate(valid_examples):
                    closest_words = [idx_word[idx.item()] for idx in closest_idxes[i]][1:]
                    print(idx_word[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")
    return model.in_embedding_layer.weight.to('cpu').data.numpy()


if __name__ == '__main__':
    # with open('vocabulary.txt', 'r') as fv:
    #     vocabulary = []
    #     while True:
    #         line = fv.readline()
    #         if not line:
    #             break
    #         vocabulary.append(line.strip())
    sentences = preprocessing.get_preprocessed_sentences()
    sorted_words = preprocessing.make_vocabulary(sentences)
    word_idx, idx_word = create_dict(sorted_words)
    sentences_in_idx = replace_words_with_idx(sentences, word_idx)
    print(len(sorted_words))

    word2vec = train(len(sorted_words), 200, sentences_in_idx, idx_word)
