import os
import random as random

import numpy

import torch
from torch.autograd import Variable


def convert2tensor(x):
    x = torch.FloatTensor(x)
    return x


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def load_top_k(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                query = parts[0].strip()
                nearest_words = parts[1].strip().split(" ")
                contexts = []
                for word in nearest_words:
                    contexts.append(word)
                dict[query] = contexts
    return dict


def load_word_clusters(langs, function_file_path, file_suffix, vocabs):
    mappings = {}
    lang_functions = {}
    for lang in langs:
        functions = {}
        word_2_idx = vocabs[lang].word2idx
        file_path = os.path.join(function_file_path, lang + file_suffix)
        line_index = 0
        function_name = ""
        with open(file_path, 'r') as f:
            for line in f:
                if line_index % 2 == 0:
                    function_name = line.strip().lower()
                elif line_index % 2 == 1:
                    parts = line.strip().lower().split('\t')
                    idx = []
                    for part in parts:
                        if part in word_2_idx:
                            id = word_2_idx[part]
                            idx.append(id)
                    functions[function_name] = idx
                line_index += 1
        lang_functions[lang] = functions

    for i in range(len(langs)):
        lang1 = langs[i]
        for j in range(len(langs)):
            lang2 = langs[j]
            if lang1==lang2:
                continue
            else:
                lang1_functions = lang_functions[lang1]
                lang2_functions = lang_functions[lang2]
                key = lang1+"-"+lang2
                dict1 = []
                dict2 = []
                for func in lang1_functions:
                    if func in lang2_functions:
                        dict1.append(lang1_functions[func])
                        dict2.append(lang2_functions[func])
                value = (dict1, dict2)
                mappings[key] = value
    return mappings


def load_functions(file, vocab, vectors):
    function_vectors = {}
    line_index = 0
    function_name = ""
    with open(file, 'r') as f:
        for line in f:
            if line_index % 2 == 0:
                function_name = line.strip().lower()
            elif line_index % 2 == 1:
                parts = line.strip().lower().split('\t')
                vec = numpy.zeros(len(vectors[0]))
                count = 0
                for part in parts:
                    if part in vocab.word2idx:
                        tmp_vec = vectors[vocab.word2idx[part]]
                        count += 1.0
                        for i in range(len(tmp_vec)):
                            vec[i] += tmp_vec[i]
                if count > 0:
                    for i in range(len(vec)):
                        vec[i] = vec[i]/count
                    function_vectors[function_name] = vec
            line_index += 1
    return function_vectors


def load_prefix_vectors(prefix_file, vocab1, vocab2, vectors1, vectors2):
    lang1_prefix_vectors = {}
    lang2_prefix_vectors = {}
    line_index = 0
    with open(prefix_file, 'r') as f:
        for line in f:
            if line_index % 3 == 0:
                name = line.strip().lower()
            elif line_index % 3 == 1:
                lang1_vec = numpy.zeros(len(vectors1[0]))
                count = 0
                low = line.strip().lower().split('\t')
                for part in low:
                    parts = part.split("##")
                    score = float(parts[2])
                    if score > 0.2 and parts[0] in vocab1.word2idx and parts[1] in vocab1.word2idx:
                        count += 1.0
                        vec_tmp1 = vectors1[vocab1.word2idx[parts[0]]]
                        vec_tmp2 = vectors1[vocab1.word2idx[parts[1]]]
                        for t in range(len(vec_tmp1)):
                            lang1_vec[t] += vec_tmp2[t]-vec_tmp1[t]
                if count > 0:
                    for t in range(len(lang1_vec)):
                        lang1_vec[t] = lang1_vec[t]/count
                    lang1_prefix_vectors[name] = lang1_vec
            elif line_index % 3 == 2:
                high = line.strip().lower().split()
                lang2_vec = numpy.zeros(len(vectors2[0]))
                count = 0
                for part in high:
                    parts = part.split("##")
                    score = float(parts[2])
                    if score > 0.2 and parts[0] in vocab2.word2idx and parts[1] in vocab2.word2idx:
                        count += 1.0
                        vec_tmp1 = vectors2[vocab2.word2idx[parts[0]]]
                        vec_tmp2 = vectors2[vocab2.word2idx[parts[1]]]
                        for t in range(len(vec_tmp1)):
                            lang2_vec[t] += vec_tmp2[t] - vec_tmp1[t]
                if count > 0:
                    for t in range(len(lang2_vec)):
                        lang2_vec[t] = lang2_vec[t] / count
                    lang2_prefix_vectors[name] = lang2_vec
            line_index += 1

    return lang1_prefix_vectors, lang2_prefix_vectors


def load_linguistic_vector(langs, file_path):
    lang_vectors = {}
    for lang in langs:
        file = os.path.join(file_path, lang + ".linguistic.vec")
        vectors = {}
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split(" ")
                word = parts[0]
                vec = []
                for item in parts[1:]:
                    d = float(item)
                    vec.append(d)
                vectors[word] = vec
        lang_vectors[lang] = vectors

    for i in range(len(langs)):
        for j in range(len(langs)):
            if i == j:
                continue
            else:
                vectors_i = lang_vectors[langs[i]]
                vectors_j = lang_vectors[langs[j]]
                num = 0
                dim = 0
                for f in vectors_i:
                    dim = len(vectors_i[f])
                    if f in vectors_j:
                        num += 1
                matrix1 = numpy.zeros((num, dim)).astype("float32", casting="same_kind")
                matrix2 = numpy.zeros((num, dim)).astype("float32", casting="same_kind")
                index = 0
                for f in vectors_i:
                    if f in vectors_j:
                        tmp1 = vectors_i[f]
                        tmp2 = vectors_j[f]
                        for m in range(len(tmp1)):
                            matrix1[index][m] = tmp1[m]
                            matrix2[index][m] = tmp2[m]
                        index += 1
                key1 = langs[i] + "#" + langs[j]
                value1 = (matrix1, matrix2)
                key2 = langs[j] + "#" + langs[i]
                value2 = (matrix2, matrix1)
                lang_vectors[key1] = value1
                lang_vectors[key2] = value2
    return lang_vectors
