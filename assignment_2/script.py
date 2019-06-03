""" Author: Dishant Mittal
	Student Id : 20710581
    Component of MSCI-641
    Created on 29/5/19
    Assignment 2
    
	https://github.com/dishant-mittal/MSCI_641/tree/master/assignment_2
    """

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

def find_test_accuracy(stopwords_include, grams):
    if stopwords_include == 'yes':
        train, train_target = shuffle(train_pos, train_neg)
        val, val_target = shuffle(val_pos, val_neg)
        test, test_target = shuffle(test_pos, test_neg)
        # train, train_target=train[0:1], train_target[0:1]
        print(train[0])
    elif stopwords_include == 'no':
        train, train_target = shuffle(train_pos_no_stopwords, train_neg_no_stopwords)
        val, val_target = shuffle(val_pos_no_stopwords, val_neg_no_stopwords)
        test, test_target = shuffle(test_pos_no_stopwords, test_neg_no_stopwords)
        # train, train_target=train[0:1], train_target[0:1]
    if grams == 'unigrams':
        vector = CountVectorizer(stop_words=[])# only unigrams
    elif grams == 'bigrams':
        vector = CountVectorizer(ngram_range=(2,2),stop_words=[])# unigrams + bigrams
    else:
        vector = CountVectorizer(ngram_range=(1,2),stop_words=[])# only bigrams
    vector.fit(train)
    counts = vector.transform(train)
    print(vector.vocabulary_)
    # print(counts.toarray())
    # print(counts.shape)
    tfidf_transformer = TfidfTransformer()
    xtrain = tfidf_transformer.fit_transform(counts)
    print(xtrain.shape)
    
    
    # HYPERPARAMETER TUNING USING VALIDATION SET
    print('tuning hyperparameters...')
    max_acc=-1
    x_max=-1
    for x in np.arange(0.1,10.0,0.2):
        clf = MultinomialNB(alpha = x, class_prior=None, fit_prior=True).fit(xtrain,train_target)
        x_new_counts = vector.transform(val)
        x_new_tfidf = tfidf_transformer.transform(x_new_counts)
        predicted = clf.predict(x_new_tfidf)
        acc = accuracy_score(val_target, predicted)
        print(acc)
        if max_acc < acc:
            max_acc = acc
            x_max = x    
    
    # TEST ACCURACY
    clf = MultinomialNB(alpha = x_max, class_prior=None, fit_prior=True).fit(xtrain,train_target)
    x_new_counts = vector.transform(test)
    x_new_tfidf = tfidf_transformer.transform(x_new_counts)
    predicted = clf.predict(x_new_tfidf)
    acc = accuracy_score(test_target, predicted)
    print('best alpha is', x_max)
    print('accuracy is',acc)
    

if __name__ == "__main__":
    input_path = sys.argv[1]


    ############### PATHS
	############### STOPWORDS INCLUDED
	test_pos=read_data("input_path/pos/test.csv")
	val_pos=read_data("input_path/pos/val.csv")
	train_pos=read_data("input_path/pos/train.csv")

	test_neg=read_data("input_path/neg/test.csv")
	val_neg=read_data("input_path/neg/val.csv")
	train_neg=read_data("input_path/neg/train.csv")


	################# STOPWORDS REMOVED
	test_pos_no_stopwords=read_data("input_path/pos/test_no_stopword.csv")
	val_pos_no_stopwords=read_data("input_path/pos/val_no_stopword.csv")
	train_pos_no_stopwords=read_data("input_path/pos/train_no_stopword.csv")

	test_neg_no_stopwords=read_data("input_path/neg/test_no_stopword.csv")
	val_neg_no_stopwords=read_data("input_path/neg/val_no_stopword.csv")
	train_neg_no_stopwords=read_data("input_path/neg/train_no_stopword.csv")


	print("STOPWORDS_REMOVED = NO, TEXT_FEATURES= UNIGRAMS")
	find_test_accuracy('no','unigrams')

	print("STOPWORDS_REMOVED = NO, TEXT_FEATURES= BIGRAMS")
	find_test_accuracy('no','bigrams')

	print("STOPWORDS_REMOVED = NO, TEXT_FEATURES= UNIGRAMS + BIGRAMS")
	find_test_accuracy('no','both')

	print("STOPWORDS_REMOVED = YES, TEXT_FEATURES= UNIGRAMS")
	find_test_accuracy('yes','unigrams')

	print("STOPWORDS_REMOVED = YES, TEXT_FEATURES= BIGRAMS")
	find_test_accuracy('yes','bigrams')

	print("STOPWORDS_REMOVED = YES, TEXT_FEATURES= UNIGRAMS + BIGRAMS")
	find_test_accuracy('yes','both')




