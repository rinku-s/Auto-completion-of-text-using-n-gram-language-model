import src.ngram as ngram


def getWordPrediction(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    """
    This function generates word prediction for the given input data

    Args:
        previous_tokens: the sequence of words as sentence from user input
        n_gram_counts: Dict of counts of n-grams to be used as denominator
        n_plus1_gram_counts: Dict of counts of (n+1)-grams to be used as numeratior
        vocabulary: List of unique words in the input data
        k: smoothing parameter, positive fractional constant
        start_with: If not None, specifies the first few letters of the next word

    Returns:
        A tuple of
          prediction: word predicted as most likely next word
          max_prob: probability of word predicted
    """
    # length of sequence of previous words
    n = len(list(n_gram_counts.keys())[0])
    # From the words that the user already typed
    # get the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]
    # Estimate the probabilities that each word in the vocabulary is the next word
    probabilities = ngram.generateProbabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)
    prediction = None
    max_prob = 0
    for word, prob in probabilities.items():
        if start_with is not None:
            if not word.startswith(start_with):
                continue
        if prob > max_prob:
            prediction = word
            max_prob = prob
    return prediction, max_prob


def getWordPredictions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    """
    This function generates list of word predictions for the given input data

    Args:
        previous_tokens: the sequence of words as sentence from user input
        n_gram_counts_list: Dict of counts of n-grams to be used as denominator for different n.
        vocabulary: List of unique words in the input data
        k: smoothing parameter, positive fractional constant
        start_with: If not None, specifies the first few letters of the next word

    Returns:
        A tuple of
          predictions: list of words predicted as most likely next word
    """
    model_counts = len(n_gram_counts_list)
    predictions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        prediction = getWordPrediction(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        predictions.append(prediction)
    return predictions
