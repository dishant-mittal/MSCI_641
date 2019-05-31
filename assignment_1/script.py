""" Author: Dishant Mittal
	Student Id : 20710581
    Component of MSCI-641
    Created on 29/5/19
    Assignment 1
    
	https://github.com/dishant-mittal/MSCI_641/tree/master/assignment_1
    """

import sys
import random
import os
import re
import numpy as np
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords

if __name__ == "__main__":
    input_path = sys.argv[1]

    """
    Tokenize the input file here
    Create train, val, and test sets
    """

    # sample_tokenized_list = [["Hello", "World", "."], ["Good", "bye"]]

    ################################################### READ DATA
    with open(input_path) as f:
    	reviews = f.readlines()


    ################################################### TOKENIZE
    pat = re.compile(r"([^\w\s])")

    #adding spaces before and after all punctuation characters
    def add_spaces(item):
        global pat
        return pat.sub(" \\1 ", item)

    reviews = [add_spaces(item) for item in reviews]
    tokenized = [item.split() for item in reviews]


    ########################## CONVERTING TOKENS INTO LOWERCASE
    ########################## AND
    ########################## REMOVING SPECIAL CHARACTERS
    pattern= r'[!#"$%&()*+/:;<=,>@\[\\\]^`{\}~\t\n]+'

    def remove_special_chars(item):
        global pattern
        temp_list = [re.sub(pattern,'', x.lower()) for x in item]
        return [i for i in temp_list if i]

    stop_words = stopwords.words('english')

    def remove_stop_words(item):
        global stop_words
        return [i for i in item if (i not in stop_words)]

    tokenized = [remove_special_chars(item) for item in tokenized]
    tokenized_no_stopwords = [remove_stop_words(item) for item in tokenized]

    ################################################## SHUFFLING (UNCOMMENT FOLLOWING IF NEED TO SHUFFLE ALSO)
    # random.shuffle(tokenized)
	# random.shuffle(tokenized_no_stopwords)


    ################################################## SPLITTING INTO TRAIN, VALIDATION AND TEST 

    train_end=int(0.8 * len(tokenized))

    validation_end=int(train_end+ 0.1 * len(tokenized))

    train_list = tokenized[0:train_end]
    val_list = tokenized[train_end:validation_end]
    test_list = tokenized[validation_end:len(tokenized)]

    train_list_no_stopword = tokenized_no_stopwords[0:train_end]
    val_list_no_stopword = tokenized_no_stopwords[train_end:validation_end]
    test_list_no_stopword = tokenized_no_stopwords[validation_end:len(tokenized)]	

    ################################################## SAVE THE FILES

    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", train_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,
               delimiter=",", fmt='%s')
