""" Author: Dishant Mittal
	Student Id : 20710581
    Component of MSCI-641
    Created on 29/5/19
    Assignment 2
    
	https://github.com/dishant-mittal/MSCI_641/tree/master/assignment_2

	this script accepts 6 parameters for input path in the following order 
	training_pos, training_neg, validation_pos, validation_neg, test_pos, test_neg	

	For example: In my personal machine I use the following path:
	python script.py ../assignment_1/pos/train.csv ../assignment_1/neg/train.csv ../assignment_1/pos/val.csv ../assignment_1/neg/val.csv ../assignment_1/pos/test.csv ../assignment_1/neg/test.csv

	Note: a row in any of the csv file is of the format: 
	['you'	 'can'	'	 't'	 'go'	 'wrong'	 'with'	 'the'	 'greatshield'	 'ultra'	 'smooth'	 '.']

    """

import sys
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def read_data(path):
    data=[]
    with open(path, 'rt') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            row[0] = row[0].replace(',', '')
            str=row[0].split()
            str[0]=str[0][1:]
            str[len(str)-1]=str[len(str)-1][:-1]
            str=[i[1:-1] for i in str] 
            data.append(' '.join(str))
        return data

def shuffle(data_pos,data_neg): 
    data = data_pos + data_neg
    merged_target = [0.0] * len(data_pos) +[1.0] * len(data_neg)
    d={'data':data,'target':merged_target}
    df = pd.DataFrame(d, columns=['data','target'])
    df = df.sample(frac=1).reset_index(drop=True)
    return list(df['data']), np.array(list(df['target']))

def find_test_accuracy(tup, grams):
    train, train_target = shuffle(tup[2], tup[5])
    val, val_target = shuffle(tup[1], tup[4])
    test, test_target = shuffle(tup[0], tup[3])
    # train, train_target=train[0:1], train_target[0:1]
    # print(train)
    if grams == 'unigrams':
        vector = CountVectorizer(stop_words=[])# only unigrams
    elif grams == 'bigrams':
        vector = CountVectorizer(ngram_range=(2,2),stop_words=[])# unigrams + bigrams
    else:
        vector = CountVectorizer(ngram_range=(1,2),stop_words=[])# only bigrams
    vector.fit(train)
    counts = vector.transform(train)
    # print(vector.vocabulary_)
    # print(counts.toarray())
    # print(counts.shape)
    tfidf_transformer = TfidfTransformer()
    xtrain = tfidf_transformer.fit_transform(counts)
    # xtrain=counts
    # print(xtrain.shape)
    
    
    # HYPERPARAMETER TUNING USING VALIDATION SET
    print('tuning hyperparameters...')
    max_acc=-1
    x_max=-1
    for x in np.arange(0.1,30.0,0.2):
        clf = MultinomialNB(alpha = x, class_prior=None, fit_prior=True).fit(xtrain,train_target)
        x_new_counts = vector.transform(val)
        x_new_tfidf = tfidf_transformer.transform(x_new_counts)
        # x_new_tfidf = x_new_counts
        predicted = clf.predict(x_new_tfidf)
        acc = accuracy_score(val_target, predicted)
        print(acc, end=' -> ', flush=True)
        if max_acc < acc:
            max_acc = acc
            x_max = x    
    
    # TEST ACCURACY
    clf = MultinomialNB(alpha = x_max, class_prior=None, fit_prior=True).fit(xtrain,train_target)
    x_new_counts = vector.transform(test)
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)
    # x_new_tfidf = x_new_counts
    predicted = clf.predict(x_new_tfidf)
    acc = accuracy_score(test_target, predicted)
    print('best alpha is', x_max)
    print('accuracy is',acc)
    

if __name__ == "__main__":
	training_pos = read_data(sys.argv[1])
	training_neg = read_data(sys.argv[2])
	validation_pos = read_data(sys.argv[3])
	validation_neg = read_data(sys.argv[4])
	test_pos = read_data(sys.argv[5])
	test_neg = read_data(sys.argv[6])
	# print(input_path)
	############### PATH
	path_tuple = (test_pos, validation_pos, training_pos,test_neg,validation_neg,training_neg)
	# print(path_tuple)

	print("TEXT_FEATURES= UNIGRAMS")
	find_test_accuracy(path_tuple,'unigrams')

	print("\nTEXT_FEATURES= BIGRAMS")
	find_test_accuracy(path_tuple,'bigrams')

	print("\nTEXT_FEATURES= UNIGRAMS + BIGRAMS")
	find_test_accuracy(path_tuple,'both')


