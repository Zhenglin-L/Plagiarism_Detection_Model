import numpy as np
import sklearn

from sklearn.feature_extraction.text import CountVectorizer


def containment_helper(a_text,s_text,n_gram):

    # instantiate an ngram counter
    counts = CountVectorizer(analyzer='word', ngram_range=(n_gram,n_gram))

    # create a dictionary of n-grams by calling `.fit`
    vocab2int = counts.fit([a_text, s_text]).vocabulary_

    # create array of n-gram counts for the answer and source text
    ngrams = counts.fit_transform([a_text, s_text])

    # row = the 2 texts and column = indexed vocab terms (as mapped above)
    # ex. column 0 = 'an', col 1 = 'answer'.. col 4 = 'text'
    ngram_array = ngrams.toarray()
    return ngram_array
    

def containment(ngram_array):
    ''' Containment is a measure of text similarity. It is the normalized, 
       intersection of ngram word counts in two texts.
       :param ngram_array: an array of ngram counts for an answer and source text.
       :return: a normalized containment value.'''
    
    # your code here
    common_min = np.sum(np.minimum(ngram_array[0],ngram_array[1]))
    containment_value = common_min/np.sum(ngram_array[0])
    
    return containment_value