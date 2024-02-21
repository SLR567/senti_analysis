import os, random, operator, sys
from collections import Counter
import numpy as np

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def readExamples(path):
    '''
    读取数据
    '''
    examples = []
    for line in open(path, encoding = 'ISO-8859-15'):
        # Format of each line: <output label (+1 or -1)> <input sentence>
        y, x = line.split(' ', 1)
        examples.append((x.strip(), int(y)))
    #print 'Read %d examples from %s' % (len(examples), path)
    return examples

def evaluatePredictor(test_text, test_label, weight, bias):
    '''
    在|examples|上测试|predictor|的性能，返回错误率
    '''
    error = 0
    for i in range(len(test_text)):
        #print(test_label[i]*(np.dot(test_text[i], weight)+bias))
        if np.dot(test_text[i], weight)+bias > 0:
            pred_label = 1
        else:
            pred_label = -1
        #if test_label[i]*(np.dot(test_text[i], weight)+bias) < 1:
            #error += 1
        if pred_label != test_label[i]:
            error += 1
    return 1.0 * error / len(test_text)
