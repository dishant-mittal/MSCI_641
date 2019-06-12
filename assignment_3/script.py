""" Author: Dishant Mittal
	Student Id : 20710581
    Component of MSCI-641
    Created on 9/6/19
    Assignment 3
    
	https://github.com/dishant-mittal/MSCI_641/tree/master/assignment_3

	this script accepts 1 parameter for input path
    """

from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import nltk
import sys


if __name__ == "__main__":
	input_path = sys.argv[1]
	# #Read data
	cols = ["reviews"]
	data_1 = pd.read_csv(input_path+'/pos.txt', names=cols, sep="\n",header=None)
	# print(len(data_1))
	data_2 = pd.read_csv(input_path+'/neg.txt', names=cols, sep="\n",header=None)
	# print(len(data_2))
	df = data_1.append(data_2)
	# print(len(data))
	df.head()

	reviews= df['reviews'].values
	#to be invoked only once:
	nltk.download('punkt')

	reviews_vec = [nltk.word_tokenize(title) for title in reviews]
	model=Word2Vec(reviews_vec, min_count=20, size=200,iter=5)

	print(model.wv.most_similar('good', topn=20))
	print(model.wv.most_similar('bad', topn=20))