import argparse
import codecs
import random

import numpy
import time

import shutil
import torch
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable

from CommonSpace import ProjLanguageContextCharLinguistic as ProjLanguage
from lang_data_loader import get_loader_bilingual_context_char
from lang_data_loader import get_loader_mono_context_char
from lang_vocab import build_vocab
from util import convert2tensor, load_linguistic_vector, load_top_k, to_var

from multilingual_eval.eval_translate import evaluate
from multilingual_eval.eval_wordsim import evaluate as evaluate_word_sim
from multilingual_eval.eval_qvec import evaluate as evaluate_qvec
from multilingual_eval.eval_cvec import evaluate as evaluate_cvec

torch.cuda.set_device(4)


def main(args):
    langs = args.langs
    embedding_path = args.mono_embedding_path
    bilingual_dict_path = args.bilingual_dict_path
    prefix = args.mono_emb_prefix
    char_prefix = args.mono_char_prefix
    model_path = args.model_path
    output_file = args.common_emb_eval
    output_file_best = args.common_emb_best
    linguistic_vec_path = args.linguistic_vec_path
    mono_dict_path = args.mono_dict_path

    # initialize model parameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    save_step = args.save_step
    log_step = save_step
    emb_size = args.word_embedding_size
    common_size = args.common_embedding_size
    kernel_num = args.kernel_num
    patience = args.patience
    max_word_length = args.max_word_length
    char_vec_size = emb_size
    filter_withs = args.filter_widths
    num_workers = args.num_workers
    top_k = args.top_k
    lg = args.lg

    # using dev sets in multilingual eval repro to select the best parameters
    eval_data_path = args.eval_data_path
    trans_path = os.path.join(eval_data_path, "word_translation/wiktionary.da+en+it.dev")
    word_sim_path = os.path.join(eval_data_path, "wordsim/en+it-mws353-dev")
    mono_sim_path = os.path.join(eval_data_path, "wordsim/EN-MEN-TR-3k")
    mono_qvec_path = os.path.join(eval_data_path, "qvec/dev-en")
    mono_qvec_cca_path = os.path.join(eval_data_path, "qvec/dev-en")
    multi_qvec_cca_path = os.path.join(eval_data_path, "qvec/dev-en-da-it")

    # create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    lang_matrixs = load_linguistic_vector(langs, linguistic_vec_path)

    context_langs = {}
    for lang in langs:
        context_langs[lang] = load_top_k(os.path.join(mono_dict_path, lang+".top50.dict"))

    # Load vocabulary wrapper.
    vocab_langs = {}
    vectors_langs = {}
    embedding_langs = {}
    char_vocab_langs = {}
    char_vectors_langs = {}
    char_embedding_langs = {}
    word2char_langs = {}
    for lang in langs:
        vocab, vectors, char_vocab, char_vectors, word_2_char = \
            build_vocab(os.path.join(embedding_path,lang+prefix), os.path.join(embedding_path, lang+char_prefix),
                        emb_size, max_word_length, char_vec_size, head=True)
        embedding = nn.Embedding(len(vectors), len(vectors[0]))
        char_embedding = nn.Embedding(len(char_vectors), len(char_vectors[0]))
        vectors_langs[lang] = vectors
        vectors = convert2tensor(vectors)
        char_vectors = convert2tensor(char_vectors)
        embedding.weight = nn.Parameter(vectors)
        embedding.weight.requires_grad = False
        char_embedding.weight = nn.Parameter(char_vectors)
        if torch.cuda.is_available():
            embedding.cuda()
            char_embedding.cuda()
        vocab_langs[lang] = vocab
        embedding_langs[lang] = embedding
        char_vocab_langs[lang] = char_vocab
        char_vectors_langs[lang] = char_vectors
        char_embedding_langs[lang] = char_embedding
        word2char_langs[lang] = word_2_char

    # Build the models
    projectors = {}
    for lang in langs:
        projector = ProjLanguage(emb_size, common_size, kernel_num, char_vec_size, filter_withs)
        if torch.cuda.is_available():
            projector.cuda()
        projectors[lang] = projector

    # Loss and Optimizer
    criterion = nn.CosineEmbeddingLoss(margin=0)
    params = []
    for lang in langs:
        params += list(projectors[lang].parameters())
    optimizer = torch.optim.Adadelta(params, lr=learning_rate)

    start = time.time()

    best_score = 0
    best_sim_score = 0
    best_model_dict= {}

    # Build data loader
    print("start to load data ... ")
    data_loader_set = get_loader_bilingual_context_char(bilingual_dict_path, langs, vocab_langs, batch_size,
                                                        word2char_langs,
                                                        shuffle=True, num_workers=num_workers, top_k=top_k)
    print("finish loading data. \nstart to train models ")
    total_step = 0
    for new_data_loader in data_loader_set:
        (lang1, lang2, data_loader) = new_data_loader
        total_step = len(data_loader)
    print("total step ", total_step)

    data_loader_mono_set = {}
    for lang in langs:
        vocab_lang = vocab_langs[lang]
        context_lang = context_langs[lang]
        word2char_lang = word2char_langs[lang]
        data_loader = get_loader_mono_context_char(os.path.join(embedding_path, lang + prefix), vocab_lang,
                                                   context_lang, word2char_lang, head=True, batch_size=batch_size,
                                                   shuffle=False, num_workers=num_workers, top_k=top_k)
        data_loader_mono_set[lang] = data_loader


    i0 = 0
    current_patience = 0
    for epoch in range(num_epochs):
        if learning_rate<0.01:
            break
        epoch_start = time.time()

        (lang01, lang02, data_loader_0) = data_loader_set[0]
        (matrix01_orig, matrix02_orig) = lang_matrixs[lang01+"#"+lang02]

        for ids01, ids02, ids01_context, ids02_context, char_ids01, char_ids02 in data_loader_0:

            # Set mini-batch dataset
            ids01 = torch.FloatTensor(ids01).long()
            ids02 = torch.FloatTensor(ids02).long()
            ids01_context = torch.FloatTensor(ids01_context).long()
            ids02_context = torch.FloatTensor(ids02_context).long()
            char_ids01 = torch.FloatTensor(char_ids01).long()
            char_ids02 = torch.FloatTensor(char_ids02).long()

            if len(matrix01_orig) > batch_size:
                gap = len(matrix01_orig)-batch_size
                rand = random.randint(0, gap)
                matrix01 = matrix01_orig[rand:rand+batch_size][:]
                matrix02 = matrix02_orig[rand:rand+batch_size][:]
            else:
                matrix01 = matrix01_orig[:][:]
                matrix02 = matrix02_orig[:][:]

            matrix01 = torch.from_numpy(matrix01)
            matrix01 = matrix01.float()
            matrix02 = torch.from_numpy(matrix02)
            matrix02 = matrix02.float()

            if torch.cuda.is_available():
                ids01 = to_var(ids01)
                ids02 = to_var(ids02)
                ids01_context = to_var(ids01_context)
                ids02_context = to_var(ids02_context)
                char_ids01 = to_var(char_ids01)
                char_ids02 = to_var(char_ids02)
                matrix01 = to_var(matrix01)
                matrix02 = to_var(matrix02)

            for langTmp in langs:
                projectors[langTmp].zero_grad()

            input01 = embedding_langs[lang01](ids01)
            input02 = embedding_langs[lang02](ids02)
            input01_context = embedding_langs[lang01](ids01_context)
            input02_context = embedding_langs[lang02](ids02_context)
            input01_context = torch.mean(input01_context, 1)
            input02_context = torch.mean(input02_context, 1)

            char_ids01_tmp = char_ids01.view(char_ids01.size(0) * char_ids01.size(1))
            char_ids02_tmp = char_ids02.view(char_ids02.size(0) * char_ids02.size(1))
            input_char01 = char_embedding_langs[lang01](char_ids01_tmp)
            input_char02 = char_embedding_langs[lang02](char_ids02_tmp)
            input_char01 = input_char01.view(char_ids01.size(0), char_ids01.size(1), -1)
            input_char02 = input_char02.view(char_ids02.size(0), char_ids02.size(1), -1)

            # Forward, Backward and Optimize
            features01, output_char01, decoded_input01, decoded_input01_context = \
                projectors[lang01].forward(input01, input01_context, input_char01)
            features02, output_char02, decoded_input02, decoded_input02_context, \
                cross_decoded_input02, cross_decoded_input02_context = \
                projectors[lang02].forward(input02, input02_context, input_char02, features01)
            features01, output_char01, decoded_input01, decoded_input01_context, \
                cross_decoded_input01, cross_decoded_input01_context = \
                projectors[lang01].forward(input01, input01_context, input_char01, features02)

            linguistic_encoded_01, linguistic_decoded_01 = projectors[lang01].forward(matrix01)
            linguistic_encoded_02, linguistic_decoded_02, cross_linguistic_decoded_02 = \
                projectors[lang02].forward(matrix02, cross_encoded=linguistic_encoded_01)
            linguistic_encoded_01, linguistic_decoded_01, cross_linguistic_decoded_01 = \
                projectors[lang01].forward(matrix01, cross_encoded=linguistic_encoded_02)

            linguistic_label0 = Variable(torch.ones(linguistic_encoded_01.size(0)))
            label00 = Variable(torch.ones(features01.size(0)))

            if torch.cuda.is_available():
                features01 = features01.cuda()
                features02 = features02.cuda()
                decoded_input01 = decoded_input01.cuda()
                decoded_input02 = decoded_input02.cuda()
                cross_decoded_input01 = cross_decoded_input01.cuda()
                cross_decoded_input02 = cross_decoded_input02.cuda()
                decoded_input01_context = decoded_input01_context.cuda()
                decoded_input02_context = decoded_input02_context.cuda()
                cross_decoded_input01_context = cross_decoded_input01_context.cuda()
                cross_decoded_input02_context = cross_decoded_input02_context.cuda()
                label00 = label00.cuda()

                output_char01 = output_char01.cuda()
                output_char02 = output_char02.cuda()

                linguistic_encoded_01 = linguistic_encoded_01.cuda()
                linguistic_encoded_02 = linguistic_encoded_02.cuda()
                linguistic_label0 = linguistic_label0.cuda()

            loss = 0

            loss += criterion(features01, features02, label00)
            loss += criterion(input01, decoded_input01, label00)
            loss += criterion(input02, decoded_input02, label00)
            loss += criterion(input01, cross_decoded_input01, label00)
            loss += criterion(input02, cross_decoded_input02, label00)

            loss += criterion(input01_context, decoded_input01_context, label00)
            loss += criterion(input01_context, cross_decoded_input01_context, label00)
            loss += criterion(input02_context, decoded_input02_context, label00)
            loss += criterion(input02_context, cross_decoded_input02_context, label00)

            char_loss = 0
            char_loss += criterion(output_char01, output_char02, label00)

            linguistic_loss = 0
            linguistic_loss += criterion(linguistic_encoded_01, linguistic_encoded_02, linguistic_label0)

            loss = loss + char_loss + lg*linguistic_loss
            loss.backward()
            optimizer.step()

            # Print log info
            if epoch>0 and i0 % log_step == 0:
                if os.path.exists(output_file):
                    os.remove(output_file)
                out = open(output_file, "w")
                for langTmp in langs:
                    data_loader = data_loader_mono_set[langTmp]
                    for i, (ids, context_ids, char_ids) in enumerate(data_loader):
                        ids = torch.FloatTensor(ids).long()
                        context_ids = torch.FloatTensor(context_ids).long()
                        char_ids = torch.FloatTensor(char_ids).long()
                        if torch.cuda.is_available():
                            ids = to_var(ids)
                            context_ids = to_var(context_ids)
                            char_ids = to_var(char_ids)

                        proj = projectors[langTmp]

                        input1 = embedding_langs[langTmp](ids)
                        input1_contexts = embedding_langs[langTmp](context_ids)

                        input1_contexts = torch.mean(input1_contexts, 1)

                        char_ids_tmp = char_ids.view(char_ids.size(0) * char_ids.size(1))
                        input_char = char_embedding_langs[langTmp](char_ids_tmp)
                        input_char = input_char.view(char_ids.size(0), char_ids.size(1), -1)

                        features, output_char, decoded_input, decoded_input_context = \
                            proj.forward(input1, input1_contexts, input_char)
                        features = torch.cat((features, output_char), 1)

                        vocab = vocab_langs[langTmp]
                        features = features.data.cpu().numpy()
                        ids = ids.data.cpu().numpy()
                        for j in range(0, len(ids)):
                            word = vocab.idx2word[ids[j]]
                            out.write(langTmp + ":" + word)
                            for m in range(len(features[j])):
                                out.write(" " + str(features[j][m]))
                            out.write("\n")
                out.close()

                mono_sim_score, mono_sim_coverate = evaluate_word_sim(mono_sim_path, output_file)
                multi_sim_score, multi_sim_coverage = evaluate_word_sim(word_sim_path, output_file)
                multi_trans_score, multi_trans_coverage = evaluate(trans_path, output_file)
                mono_qvec_score, mono_qvec_coverate = evaluate_qvec(mono_qvec_path, output_file)
                multi_qvec_score, multi_qvec_coverate = evaluate_qvec(multi_qvec_cca_path, output_file)
                mono_cvec_score, mono_cvec_coverate = evaluate_cvec(mono_qvec_cca_path, output_file)
                multi_cvec_score, multi_cvec_coverate = evaluate_cvec(multi_qvec_cca_path, output_file)

                score = mono_sim_score + multi_sim_score + multi_trans_score + \
                        mono_qvec_score + mono_cvec_score + multi_cvec_score

                print("mono_sim: %.4f, multi_sim: %.4f, multi_trans: %.4f, mono_qvec: %.4f, multi_qvec: %.4f, "
                      "mono_cvec: %.4f, multi_cvec: %.4f" %
                      (mono_sim_score, multi_sim_score, multi_trans_score, mono_qvec_score,
                       multi_qvec_score, mono_cvec_score, multi_cvec_score))

                print("\n")

                if score > best_score:
                    shutil.copyfile(output_file, output_file_best)
                    current_patience = 0
                    best_score = score
                    for tmp in langs:
                        best_model_dict[tmp] = projectors[tmp].state_dict()
                else:
                    current_patience += 1

                if current_patience > patience:
                    learning_rate = learning_rate * 0.5
                    current_patience = 0

                epoch_end = time.time()
                epoch_time = epoch_end - epoch_start
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Best Score: %.4f, Best WordSim: %.4f, Learning_Rate: '
                      '%.4f, CurrentPatience: %.4f, Perplexity: %5.4f, Time: %d' %
                      (epoch, num_epochs, i0, total_step, loss.data[0], best_score, best_sim_score,
                       learning_rate, current_patience, np.exp(loss.data[0]), epoch_time))

                epoch_start = time.time()

                # Save the models
            if (epoch + 1) % save_step == 0:
                for tmp in langs:
                    torch.save(projectors[tmp].state_dict(),
                               os.path.join(model_path, tmp + '-encoder-%d-%d.pkl' % (epoch + 1, i0 + 1)))
            i0 += 1

    end = time.time()
    all_time = end - start
    print('Overall training time %d' % all_time)
    for lang in langs:
        torch.save(best_model_dict[lang], os.path.join(model_path, lang+'-best-encoder.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # here list the parameters for optimizing the model
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--word_embedding_size", type=int, default=512)
    parser.add_argument("--common_embedding_size", type=int, default=560)
    parser.add_argument("--kernel_num", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_word_length", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--lg", type=float, default=0.2)
    parser.add_argument("--filter_widths", nargs="+", type=int, help="list of integer filter widths")

    # here list the dev data path
    parser.add_argument("--eval_data_path", type=str, default='./multilingual_eval/eval-data/',
                        help="path of multilingual eval data path")

    # here list the path for all input
    parser.add_argument("--langs", nargs="+", type=str, help="list of languages to be projected into the common space")
    parser.add_argument("--mono_embedding_path", type=str, default="./data/mono-embedding/",
                        help="path of monolingual embeddings")
    parser.add_argument("--mono_emb_prefix", type=str, default=".size-512.window-5.iter-10.vec",
                        help="prefix of monolingual word embeddings")
    parser.add_argument("--mono_char_prefix", type=str, default=".char.size-512.window-5.iter-10.vec",
                        help="prefix of monolingual character embeddings")
    parser.add_argument("--model_path", type=str, default="./models/", help="path for saving models")
    parser.add_argument("--common_emb_eval", type=str, default="./data/multi-embedding/trilingual.eval",
                        help="path to save the multilingual embeddings after each batch training")
    parser.add_argument("--common_emb_best", type=str, default="./data/multi-embedding/trilingual.best",
                        help="path to save the best multilingual embeddings")
    parser.add_argument("--linguistic_vec_path", type=str, default="./data/linguistic_vectors/",
                        help="path of the linguistic feature vectors")
    parser.add_argument("--bilingual_dict_path", type=str, default="./data/bilingual-top-dict/",
                        help="path of the top k words for each pair of bilingual word alignment")
    parser.add_argument("--mono_dict_path", type=str, default="./data/mono-top-dict/",
                        help="path of the top k words for each language")

    args = parser.parse_args()
    main(args)