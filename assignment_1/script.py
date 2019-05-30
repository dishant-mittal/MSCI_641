""" Author: Dishant Mittal
    Component of MSCI-641
    Created on 29/5/19
    Assignment 1
    
	https://github.com/dishant-mittal/MSCI_641/tree/master/assignment_1
    """

import sys
import pandas as pd
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
    cols = ["reviews"]
    df = pd.read_csv(input_path, names=cols, sep="\n",header=None)
    # df.head()


    ################################################### TOKENIZE
    pat = re.compile(r"([^\w\s])")

    #adding spaces before and after all punctuation characters
    def add_spaces(item):
        global pat
        return pat.sub(" \\1 ", item)

    df['reviews']=df['reviews'].apply(add_spaces)
    df['tokenized'] = df["reviews"].str.split(" ", expand = False)
    # df.head()


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

    df['tokenized'] = df['tokenized'].apply(remove_special_chars)
    df['tokenized_no_stopwords'] = df['tokenized'].apply(remove_stop_words)

    # df.head()

    ################################################## SHUFFLING (UNCOMMENT FOLLOWING IF NEED TO SHUFFLE ALSO)
    #shuffle using pandas inbuilt method
    # df = df.sample(frac=1).reset_index(drop=True)

    #OR

    # shuffle_dataset (my own shuffle code from scratch) uncomment the following method to test
    # def shuffle_in_place(array):
    #     array_len = len(array)
    #     assert array_len > 2, 'This list is very short to shuffle'
    #     for index in range(array_len):
    #         swap = random.randrange(array_len - 1)
    #         swap += swap >= index
    #         array.iloc[index], array.iloc[swap] = array.iloc[swap], array.iloc[index]
            
    # # arr=[1,2,3,4,5,6]
    # shuffle_in_place(df)
    # df.head()


    ################################################## SPLITTING INTO TRAIN, VALIDATION AND TEST 

    train_end=int(0.8 * len(df))

    validation_end=int(train_end+ 0.1 * len(df))

    train_set = df[0:train_end]
    validation_set = df[train_end:validation_end+1]
    test_set = df[validation_end+1:len(df)]

    #save_lists
    train_list= list(train_set['tokenized'])
    val_list= list(validation_set['tokenized'])
    test_list= list(test_set['tokenized'])

    train_list_no_stopword = list(train_set['tokenized_no_stopwords'])
    val_list_no_stopword = list(validation_set['tokenized_no_stopwords'])
    test_list_no_stopword = list(test_set['tokenized_no_stopwords'])    

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
