import numpy as np
import pandas as pd
from src.constants import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN

def countNGrams(data, n, start_token=START_TOKEN, end_token=END_TOKEN):
    """
    This function is used for counting the n-grams in a sequence. n-grams are calculated for
    n = length of the sequence.
    ***
    Args:
        data: List of lists of words
        n: number of words in a sequence
    ***
    Returns:
        n_grams: dict with key as a tuple of n-words and value as its frequency in the data
    """
    # Initialize dictionary of n-grams and their counts
    n_grams = {}
    # Go through each sentence in the data
    for sentence in data:
        # prepend start token n times, and  append <E> one time
        sentence = n * [start_token] + sentence + [end_token]
        # convert list to tuple
        # So that the sequence of words can be used as a key in the dictionary
        sentence = tuple(sentence)
        # Use 'i' to indicate the start of the n-gram from index 0 to the last index where the end of the n-gram
        # is within the sentence.
        for i in range(len(sentence) - n + 1):
            # Get the n-gram from i to i+n
            n_gram = sentence[i: i + n]
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    return n_grams


def generateProbability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    This function estimate the probability of a next word using the n-gram counts with k-smoothing.

    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter

    Returns:
        probability:probability of the next word
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)
    # Set the denominator
    # If the previous n-gram exists in the dictionary of n-gram counts,
    # Get its count.  Otherwise set the count to zero
    # Use the dictionary that has counts for n-grams
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + (k * vocabulary_size)
    # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    numerator = n_plus1_gram_count + k
    probability = numerator / denominator
    return probability


def generateProbabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    """
    This function estimate the probabilities of next words using the n-gram counts with k-smoothing.

    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of (n+1)-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter

    Returns:
        probabilities: A dictionary mapping from next words to the probability.
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)
    # add <E> <unk> to the vocabulary
    # <S> is not needed since it should not appear as the next word
    vocabulary = vocabulary + [END_TOKEN, UNKNOWN_TOKEN]
    vocabulary_size = len(vocabulary)
    probabilities = {}
    for word in vocabulary:
        probability = generateProbability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        probabilities[word] = probability
    return probabilities


def generateCountMatrix(n_plus1_gram_counts, vocab):
    """
    This function generates a matrix of the counts of n-gram.

    Args:
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words

    Returns:
        count_matrix: matrix of counts of n-gram
    """
    vocab = vocab + [END_TOKEN, UNKNOWN_TOKEN]
    n_grams = []

    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))

    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    col_index = {word: j for j, word in enumerate(vocab)}

    count_matrix = np.zeros((len(n_grams), len(vocab)))

    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[:-1]
        word = n_plus1_gram[-1]
        if word not in vocab:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocab)
    return count_matrix


def generateProbabilityMatrix(n_plus1_gram_counts, vocab, k):
    """
    This function generates a matrix of the probababilities of n-gram.

    Args:
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: smoothing parameter

    Returns:
        prob_matrix: matrix of probabilities of n-gram
    """
    count_matrix = generateCountMatrix(n_plus1_gram_counts, vocab)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix
