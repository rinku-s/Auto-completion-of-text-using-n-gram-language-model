import math
import random

def splitData(tokenized_sentences):
    """
    This function splits first shuffles the data and then splits it into train set and test set.
    ***
    Args:
    tokenized_sentences: List of lists of strings
    ***
    Returns:
    train_data: list of train data
    test_data: list of test data
    """
    random.seed(10)
    random.shuffle(tokenized_sentences)
    train_data_size = int(len(tokenized_sentences) * 0.8)
    train_data = tokenized_sentences[0:train_data_size]
    test_data = tokenized_sentences[train_data_size:]
    return train_data, test_data