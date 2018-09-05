import codecs
import torch
import torch.utils.data as data
import os


class BilingualAlignmentContextChar(data.Dataset):
    def __init__(self, dict_file, lang1, lang2, vocab_langs, word2char_langs, top_k):
        self.dict_file = dict_file
        self.lang1 = lang1
        self.lang2 = lang2
        self.vocab_langs = vocab_langs
        self.word2char_langs = word2char_langs
        self.top_k = top_k
        dict1, dict2, dict1_context, dict2_context, char_dict1, char_dict2 = \
            self.__load_dict__(vocab_langs[lang1], vocab_langs[lang2], dict_file,
                               word2char_langs[lang1], word2char_langs[lang2], top_k)
        self.dict1 = dict1
        self.dict2 = dict2
        self.dict1_context = dict1_context
        self.dict2_context = dict2_context
        self.char_dict1 = char_dict1
        self.char_dict2 = char_dict2
        self.ids = list(self.dict1.keys())

    def __load_dict__(self, vocab1, vocab2, dict_file, word2char_lang1, word2char_lang2, top_k):
        dict1 = {}
        dict2 = {}
        dict1_context = {}
        dict2_context = {}
        char_dict1 = {}
        char_dict2 = {}
        index = 0
        with open(dict_file, 'r') as f:
            for line in f:
                parts = line.lower().strip().split("\t")
                word1 = parts[0][3:].strip()
                word2 = parts[1][3:].strip()
                if len(parts) == 4:
                    word1_context = parts[2].strip().split(" ")
                    word2_context = parts[3].strip().split(" ")

                    if word1 in vocab1.word2idx and word2 in vocab2.word2idx:
                        word_index1 = vocab1.word2idx[word1]
                        word_index2 = vocab2.word2idx[word2]

                        char_index1 = word2char_lang1[word_index1]
                        char_index2 = word2char_lang2[word_index2]

                        word1_context_tmp = []
                        word2_context_tmp = []
                        count1 = 0
                        count2 = 0
                        for tmp1 in word1_context:
                            tmp1 = tmp1[3:].strip()
                            if count1 < top_k and tmp1 in vocab1.word2idx:
                                word1_context_tmp.append(vocab1.word2idx[tmp1])
                                count1 += 1
                        for tmp2 in word2_context:
                            tmp2 = tmp2[3:].strip()
                            if count2 < top_k and tmp2 in vocab2.word2idx:
                                word2_context_tmp.append(vocab2.word2idx[tmp2])
                                count2 += 1
                        if len(char_index1) > 0 and len(char_index2) > 0 and len(word1_context_tmp) > 0 \
                                and len(word2_context_tmp) > 0:
                            dict1[index] = word_index1
                            dict2[index] = word_index2
                            char_dict1[index] = char_index1
                            char_dict2[index] = char_index2
                            dict1_context[index] = word1_context_tmp
                            dict2_context[index] = word2_context_tmp
                            index += 1
        return dict1, dict2, dict1_context, dict2_context, char_dict1, char_dict2

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        id1 = self.dict1[index]
        id2 = self.dict2[index]

        id1_context = self.dict1_context[index]
        id2_context = self.dict2_context[index]

        char_id1 = self.char_dict1[index]
        char_id2 = self.char_dict2[index]

        return id1, id2, id1_context, id2_context, char_id1, char_id2

    def __len__(self):
        return len(self.ids)


def collate_fn_context_char(data):
    """Creates mini-batch tensors from the list of tuples
    """
    ids1, ids2, ids1_context, ids2_context, char_ids1, char_ids2 = zip(*data)

    lengths1 = [len(ids1_context_tmp) for ids1_context_tmp in ids1_context]
    ids1_context_targets = torch.zeros(len(ids1_context), max(lengths1)).float()
    for i, ids1_context_tmp in enumerate(ids1_context):
        end = lengths1[i]
        ids1_context_tmp = torch.FloatTensor(ids1_context_tmp)
        ids1_context_targets[i, :end] = ids1_context_tmp[:end]

    lengths2 = [len(ids2_context_tmp) for ids2_context_tmp in ids2_context]
    ids2_context_targets = torch.zeros(len(ids2_context), max(lengths2)).float()
    for i, ids2_context_tmp in enumerate(ids2_context):
        end = lengths2[i]
        ids2_context_tmp = torch.FloatTensor(ids2_context_tmp)
        ids2_context_targets[i, :end] = ids2_context_tmp[:end]

    lengths1 = [len(char_ids1_tmp) for char_ids1_tmp in char_ids1]
    char_ids1_targets = torch.zeros(len(char_ids1), max(lengths1)).float()
    for i, char_ids1_tmp in enumerate(char_ids1):
        end = lengths1[i]
        char_ids1_tmp = torch.from_numpy(char_ids1_tmp).float()
        char_ids1_targets[i, :end] = char_ids1_tmp[:end]

    lengths2 = [len(char_ids2_tmp) for char_ids2_tmp in char_ids2]
    char_ids2_targets = torch.zeros(len(char_ids2), max(lengths2)).float()
    for i, char_ids2_tmp in enumerate(char_ids2):
        end = lengths2[i]
        char_ids2_tmp = torch.from_numpy(char_ids2_tmp).float()
        char_ids2_targets[i, :end] = char_ids2_tmp[:end]

    return ids1, ids2, ids1_context_targets, ids2_context_targets, char_ids1_targets, char_ids2_targets


def get_loader_bilingual_context_char(dict_path, langs, vocab_langs, batch_size, word2char_langs,
                                      shuffle, num_workers, top_k):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # Data loader for bilingual dictionary
    data_loaders = []

    visited_file = []
    for i in range(0, len(langs)):
        for j in range(0, len(langs)):
            lang1 = langs[i]
            lang2 = langs[j]
            dict_file1 = os.path.join(dict_path, lang1 + "." + lang2 + ".dict.top50")
            dict_file2 = os.path.join(dict_path, lang2 + "." + lang1 + ".dict.top50")
            if os.path.exists(dict_file1) and dict_file1 not in visited_file:
                visited_file.append(dict_file1)
                data_set = BilingualAlignmentContextChar(dict_file1, lang1, lang2, vocab_langs, word2char_langs, top_k)
                data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle,
                                                          num_workers=num_workers,
                                                          collate_fn=collate_fn_context_char)
                new_data_loader = (lang1, lang2, data_loader)
                data_loaders.append(new_data_loader)

            elif os.path.exists(dict_file2) and dict_file2 not in visited_file:
                visited_file.append(dict_file2)
                data_set = BilingualAlignmentContextChar(dict_file2, lang2, lang1, vocab_langs, word2char_langs, top_k)
                data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle,
                                                          num_workers=num_workers,
                                                          collate_fn=collate_fn_context_char)
                new_data_loader = (lang2, lang1, data_loader)
                data_loaders.append(new_data_loader)

    return data_loaders


class MonolingualContextChar(data.Dataset):
    def __init__(self, word2vec_file, vocab_lang, context_lang, word2char_lang, head, top_k):
        self.vocab_lang = vocab_lang
        self.context_lang = context_lang
        self.top_k = top_k
        dict1, context_dict1, char_dict1 = self.__load_dict__(word2vec_file, vocab_lang, context_lang,
                                                              word2char_lang, head, top_k)
        self.dict1 = dict1
        self.context_dict1 = context_dict1
        self.char_dict1 = char_dict1
        self.ids = list(self.dict1.keys())

    def __load_dict__(self, word2vec_file, vocab_lang, context_lang, word2char_lang, head, top_k):
        dict1 = {}
        context_dict = {}
        char_dict1 = {}
        words = {}
        index = 0
        count = 0
        top_k = top_k
        with open(word2vec_file, 'r') as f:
            for line in f:
                if head is True and count == 0:
                    count = 1
                    continue
                else:
                    parts = line.split(" ")
                    word1 = parts[0].strip()
                    if word1 in vocab_lang.word2idx and word1 in context_lang:
                        word_index1 = vocab_lang.word2idx[word1]
                        dict1[index] = word_index1
                        char_index1 = word2char_lang[word_index1]
                        char_dict1[index] = char_index1

                        context_tmp_words = context_lang[word1]
                        context_words = []
                        count1 = 0
                        for tmp1 in context_tmp_words:
                            if count1 < top_k and tmp1 in vocab_lang.word2idx:
                                context_words.append(vocab_lang.word2idx[tmp1])
                                count1 += 1
                        if len(context_words)>0 and len(word1)>0:
                            context_dict[index] = context_words
                            words[index] = word1
                            index += 1
        return dict1, context_dict, char_dict1

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        id1 = self.dict1[index]
        context_id1 = self.context_dict1[index]
        char_id1 = self.char_dict1[index]
        return index, id1, context_id1, char_id1

    def __len__(self):
        return len(self.ids)


def collate_fn_mono_context_char(data):
    """Creates mini-batch tensors from the list of tuples
    """
    idxs, ids, ids1_context, char_ids1 = zip(*data)
    lengths1 = [len(ids1_context_tmp) for ids1_context_tmp in ids1_context]
    ids1_context_targets = torch.zeros(len(ids1_context), max(lengths1)).float()
    for i, ids1_context_tmp in enumerate(ids1_context):
        end = lengths1[i]
        ids1_context_tmp = torch.FloatTensor(ids1_context_tmp)
        ids1_context_targets[i, :end] = ids1_context_tmp[:end]

    lengths1 = [len(char_ids1_tmp) for char_ids1_tmp in char_ids1]
    char_ids1_targets = torch.zeros(len(char_ids1), max(lengths1)).float()
    for i, char_ids1_tmp in enumerate(char_ids1):
        end = lengths1[i]
        char_ids1_tmp = torch.from_numpy(char_ids1_tmp).float()
        char_ids1_targets[i, :end] = char_ids1_tmp[:end]

    return ids, ids1_context_targets, char_ids1_targets


def get_loader_mono_context_char(word2vec_file, vocab_lang, context_lang, word2char_lang, head, batch_size,
                                 shuffle, num_workers, top_k):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    data_set = MonolingualContextChar(word2vec_file, vocab_lang, context_lang, word2char_lang, head, top_k)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_mono_context_char)
    return data_loader
