#!/usr/bin/python

import random
import collections
import numpy as np
import math
import sys
from util import *
import gensim.downloader
import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
from fse.models import SIF
from fse import IndexedList
from fse.models import Average
import matplotlib.pyplot as plt


#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

def preprocessing(corpus):
    stemmer = SnowballStemmer("english")
    tokenized_corpus = [word_tokenize(sent) for sent in corpus]
    stemmed = []
    for sent in tokenized_corpus:
        stemmed_sent=[stemmer.stem(word) for word in sent]
        stemmed.append(stemmed_sent)
    
    pos = [pos_tag(sent) for sent in stemmed]
    
    lemmatizer = WordNetLemmatizer()
    result = []
    result_string = []
    for sent in pos:
        temp = []
        for word, tag in sent:
            if tag.startswith('J'):
                temp.append(lemmatizer.lemmatize(word, pos = nltk.corpus.wordnet.ADJ))
            elif tag.startswith('V'):
                temp.append(lemmatizer.lemmatize(word, pos = nltk.corpus.wordnet.VERB))
            elif tag.startswith('N'):
                temp.append(lemmatizer.lemmatize(word, pos = nltk.corpus.wordnet.NOUN))
            elif tag.startswith('R'):
                temp.append(lemmatizer.lemmatize(word, pos = nltk.corpus.wordnet.ADV))
            else:
                temp.append(lemmatizer.lemmatize(word, pos = nltk.corpus.wordnet.NOUN))
        result.append(temp)
        result_string.append(' '.join(temp))
    
    return result_string


def sif_embeddings(sentences, model, alpha=1e-3):
    """Compute the SIF embeddings for a list of sentences
    Parameters
    ----------
    sentences : list
        The sentences to compute the embeddings for
    model : `~gensim.models.base_any2vec.BaseAny2VecModel`
        A gensim model that contains the word vectors and the vocabulary
    alpha : float, optional
        Parameter which is used to weigh each individual word based on its probability p(w).
    Returns
    -------
    numpy.ndarray 
        SIF sentence embedding matrix of dim len(sentences) * dimension
    """
    
    vlookup = model.wv.key_to_index  # Gives us access to word index and count
    vectors = model.wv        # Gives us access to word vectors
    size = model.vector_size  # Embedding size
    
    Z = 0
    for k in vlookup:
        vector = model.wv.get_vector(k)
        Z += np.count_nonzero(vector)
        #Z += model.wv.get_vector(k).count_nonzero() # Compute the normalization constant Z
    
    output = []
    
    # Iterate all sentences
    for s in sentences:
        count = 0
        v = np.zeros(size) # Summary vector
        # Iterare all words
        for w in s:
            # A word must be present in the vocabulary
            if w in vlookup:
                vvvv = model.wv.get_vector(w)
                cccc= np.count_nonzero(vvvv)
                for i in range(size):
                    v[i] += ( alpha / (alpha + (cccc / Z))) * vectors[w][i]
                count += 1 
                
        if count > 0:
            for i in range(size):
                v[i] *= 1/count
        output.append(v)
    return np.vstack(output)


def extractFeatures(corpus, mode):
    """
    从语料库中提取特征
    @param list of list of strings corpus: 
    @return vector: feature vector representations of corpus.
    """
    noise = stopwords.words("english")
    corpus = preprocessing(corpus)
    # BEGIN_YOUR_CODE 
    if mode == 'BOW':
    # bag of word
        #extractor = CountVectorizer()
        #freq_info = extractor.fit_transform(corpus)
        #freq_matrix = freq_info.toarray()
        #tf_idf = TfidfTransformer()
        #vectors = tf_idf(freq_matrix).toarray()
        
        tf_idf1 = TfidfVectorizer(decode_error = 'ignore', stop_words = noise)
        vectors1 = tf_idf1.fit_transform(corpus).toarray()
        return vectors1
    
    elif mode == 'Bigram':
        tf_idf1 = TfidfVectorizer(decode_error = 'ignore', stop_words = noise, ngram_range = (2, 2))
        vectors1 = tf_idf1.fit_transform(corpus).toarray()
        return vectors1
    
    elif mode == 'Trigram':
        tf_idf1 = TfidfVectorizer(decode_error = 'ignore', stop_words = noise, ngram_range = (3, 3))
        vectors1 = tf_idf1.fit_transform(corpus).toarray()
        return vectors1
    
    elif mode == 'Combo':
        tf_idf1 = TfidfVectorizer(decode_error = 'ignore', stop_words = noise, ngram_range = (1, 3))
        vectors1 = tf_idf1.fit_transform(corpus).toarray()
        return vectors1
    
    elif mode == 'Glove':
        model = gensim.downloader.load('glove-twitter-50')
        embed_size = model.wv['i'].shape[0]
        corpus = [sent.split() for sent in corpus]
        avg_embed = []
        for sent in corpus:
            avg_vec = np.zeros(embed_size)
            len_sent = 0
            for word in sent:
                try:
                    len_sent += 1
                    vector = model.wv[word]
                except:
                    continue
                avg_vec += vector
            avg_embed.append(avg_vec / len_sent)
        avg_embed = np.array(avg_embed)
            
        sif_embed = sif_embeddings(corpus, model)
        return sif_embed
    
    else:
        corpus = [sent.split() for sent in corpus]
        model = gensim.models.Word2Vec(corpus, window=20, min_count=5, vector_size=50)
        embed_size = model.wv['i'].shape[0]
        avg_embed = []
        for sent in corpus:
            avg_vec = np.zeros(embed_size)
            len_sent = 0
            for word in sent:
                try:
                    len_sent += 1
                    vector = model.wv[word]
                except:
                    continue
                avg_vec += vector
            avg_embed.append(avg_vec / len_sent)
        avg_embed = np.array(avg_embed)

        sif_embed = sif_embeddings(corpus, model)
        return sif_embed
                         
    # END_YOUR_CODE
    
    
def plot_loss(loss, plot_type, mode, eta):
    fig, ax = plt.subplots(1,1)
    if plot_type== 'train':
        ax.set_title("Training Loss vs Iterations")
        ax.set_xlabel('Iters')
        ax.set_ylabel("Training Loss")
        ax.plot(loss)
        fig.savefig('training_loss_vs_iters_'+ mode + str(eta) + '.png')
    if plot_type== 'test':
        ax.set_title('Test Error vs Iterations')
        ax.set_xlabel('Iters')
        ax.set_ylabel('Validation Loss')
        ax.plot(loss)
        fig.savefig('test_error_vs_iters_' + mode + str(eta) + '.png')
    plt.close()
    
def learnPredictor(train_embed, test_embed, train_label, test_label, numIters, eta, reg):
    '''
    给定训练数据和测试数据，特征提取器|featureExtractor|、训练轮数|numIters|和学习率|eta|，
    返回学习后的权重weights
    你需要实现随机梯度下降优化权重
    '''
    
    #train_text, val_text, train_l, val_l = train_test_split(train_embed, train_label, test_size = 0.1, random_state = 31)
    num_train = train_embed.shape[0]
    num_features = train_embed.shape[1]
    
    weight = np.zeros(num_features)
    bias = 0
    stop_criterion = 0.00001
    min_test_error = float('inf')
    
    training_loss = []
    test_loss =[]
    test_error_list = []
    temp = 0
    early_stop = 5
    stop = 0
    for t in range(numIters):
        i = random.randint(0, num_train-1)
        example = train_embed[i,:]
        #print(train_label[i])
        pred_score = train_label[i] * (np.dot(example, weight) + bias)
        loss = max(0, 1-pred_score)
        temp += loss
        #print(weight)
        #print(bias)
        if pred_score >= 1:
            #print('true')
            weight -= eta * (reg * weight)
            #continue
        else:
            weight -= eta * (reg * weight - np.dot(example, train_label[i]))
            #weight -= eta * (-np.dot(example, train_label[i]))
            #bias -= eta * train_label[i]
        
        if t%5000 == 0:
            if temp != 0 and t != 0:
                training_loss.append(temp/5000)
                print('training loss at {} is: {}'.format(t, temp/5000))
                temp = 0
            train_error = evaluatePredictor(train_embed, train_label, weight, bias)
            print('training error at {} is: {}'.format(t, train_error))

            test_error = evaluatePredictor(test_embed, test_label, weight, bias)
            print('test error at {} is: {}'.format(t, test_error))  
            
            if test_error > min_test_error:
                stop += 1
            else:
                min_test_error = test_error
                stop = 0
            if stop > early_stop:
                break
            curr_loss = 0
            for i, example in enumerate(test_embed):
                curr_loss += max(0, 1-test_label[i]*(np.dot(example, weight)+bias))
       
            curr_loss = curr_loss/len(test_embed)
            print('current test loss is: {}'.format(curr_loss))
            #if (prev_loss - curr_loss) < prev_loss * stop_criterion:
                #return weight, bias
            #else:
                #prev_loss = curr_loss
            
            test_loss.append(curr_loss)
            test_error_list.append(test_error)
           
    #plot_loss(training_loss, 'train')
    #plot_loss(test_error_list, 'test')
        
    return weight, bias, training_loss, test_error_list
        
        
                
    # # BEGIN_YOUR_CODE 
    # raise Exception("Not implemented yet")
    # # END_YOUR_CODE
    # return weights


