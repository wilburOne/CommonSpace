import random
import numpy as np


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(file, char_emb_file, emb_size, max_word_length, char_vec_dim, head):
    pretrain_char_vecs = {}
    with open(char_emb_file, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            vec = []
            for item in parts[1:]:
                vec.append(float(item))
            pretrain_char_vecs[parts[0]] = vec
    vectors = []
    char_vectors = []
    vocab = Vocabulary()
    char_vocab = Vocabulary()
    count = 0
    with open(file, 'r') as f:
        for line in f:
            if head is True and count == 0:
                count += 1
                continue
            else:
                line = line.strip()
                parts = line.split(" ")
                if len(parts) == emb_size+1:
                    word = parts[0].strip()
                    vocab.add_word(word)
                    if len(word) > max_word_length:
                        max_word_length = len(word)
                    vec = []
                    for item in parts[1:]:
                        vec.append(float(item))
                    vectors.append(vec)

    word_2_char = np.zeros([len(vocab.word2idx), max_word_length], dtype=np.int32)
    char_vocab.add_word("PAD")

    for i in range(0, len(vocab.word2idx)):
        word = vocab.idx2word[i]
        for j in range(len(word)):
            if word[j] not in char_vocab.word2idx:
                char_vocab.add_word(word[j])
            char_index = char_vocab.word2idx[word[j]]
            word_2_char[i][j] = char_index

    for i in range(0, len(char_vocab.idx2word)):
        char_vec_tmp = []
        char_str = char_vocab.idx2word[i]
        if i==0:
            for m in range(char_vec_dim):
                char_vec_tmp.append(0)  # random.rand(-0.5, 0.5)
        else:
            if char_str in pretrain_char_vecs:
                char_vec_tmp = pretrain_char_vecs[char_str]
            else:
                for m in range(char_vec_dim):
                    char_vec_tmp.append(random.uniform(-0.5, 0.5))
        char_vectors.append(char_vec_tmp)

    return vocab, vectors, char_vocab, char_vectors, word_2_char
