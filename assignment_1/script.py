""" Author: Dishant Mittal
    Component of MSCI-641
    Created on 29/5/19
    Assignment 1

    

    
    """




import sys

if __name__ == "__main__":
    input_path = sys.argv[1]

    """
    Tokenize the input file here
    Create train, val, and test sets
    """

    # sample_tokenized_list = [["Hello", "World", "."], ["Good", "bye"]]

    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", train_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,
               delimiter=",", fmt='%s')