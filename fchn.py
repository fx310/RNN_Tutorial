# -*- coding=utf-8 -*-
import sys
import math
import codecs
import numpy as np
import timeit
from frnn import RNNNumpy

def generate_sentence(model,word_to_index,index_to_word,sentence_start_token,sentence_end_token,unknown_token):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs,s = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

def main():
    if len(sys.argv) != 7:
        print "usage: python fchn.py vocab input.txt model.npy hidden bptt (train|test|gens|gchk)"
        sys.exit(0)

    hidden, bptt = int(sys.argv[4]), int(sys.argv[5])

    # vocab
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    unknown_token = "UNKNOWN_TOKEN"
    vocab = []
    sentences = []
    vocab.append(sentence_start_token)
    vocab.append(sentence_end_token)
    vocab.append(unknown_token)
    vocab_size = 3

    file = codecs.open(sys.argv[1], "rb", "utf-8")
 
    while 1:
        lines = file.readlines(10000)
        if not lines:
            break
        for sent in lines:
            sent = sent.rstrip()
            vocab.append(sent)
            vocab_size = vocab_size + 1
            # print sent.encode('utf8')
    file.close()

    word_to_index = dict([(w,i) for i,w in enumerate(vocab)])
    index_to_word = dict([(i,w) for i,w in enumerate(vocab)])
    
    file = codecs.open(sys.argv[2], "rb", "utf-8")
 
    while 1:
        lines = file.readlines(10000)
        if not lines:
            break
        for sent in lines:
            sent = sent.rstrip().split(' ')
            sent_token = [w if w in word_to_index else unknown_token for w in sent]
            sent_token.append(sentence_end_token)
            sent_token.insert(0,sentence_start_token)
            sentences.append(sent_token)
    file.close()

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])
    # print X_train[0]
    # print y_train[0]

    if sys.argv[6] == 'train':
        np.random.seed(10)
        model = RNNNumpy(word_dim=vocab_size, hidden_dim=hidden, bptt_truncate=bptt)
        model.train_with_sgd(X_train, y_train, nepoch=20, evaluate_loss_after=1)
        model.save(sys.argv[3])
    
    if sys.argv[6] == 'test':
        np.random.seed(10)
        model = RNNNumpy(word_dim=vocab_size)
        model.load(sys.argv[3])
        print "Actual loss: %f" % model.calculate_loss(X_train, y_train)
    
    if sys.argv[6] == 'gchk':
        grad_check_vocab_size = 100
        np.random.seed(10)
        model = RNNNumpy(grad_check_vocab_size, 10, 3)
        model.gradient_check([0,1,2,3], [1,2,3,4])

    if sys.argv[6] == 'gens':
        np.random.seed(10)
        model = RNNNumpy(word_dim=vocab_size)
        model.load(sys.argv[3])

        num_sentences = 10
        senten_min_length = 5

        for i in range(num_sentences):
            sent = []
            # We want long sentences, not sentences with one or two words
            while len(sent) < senten_min_length:
                sent = generate_sentence(model,word_to_index,index_to_word,sentence_start_token,sentence_end_token,unknown_token)
            print " ".join(sent)

if __name__ == "__main__":
    main()

