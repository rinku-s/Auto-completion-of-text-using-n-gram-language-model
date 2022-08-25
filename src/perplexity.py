import src.ngram as ngram
from src.constants import START_TOKEN, END_TOKEN


def getPerplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    This function calculate perplexity for a list of sentences.

    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of (n+1)-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant

    Returns:
        perplexity: Perplexity score of given input
    """
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])
    # prepend <s> and append <e>
    sentence = [START_TOKEN] * n + sentence + [END_TOKEN]
    sentence = tuple(sentence)
    # length of sentence (after adding <s> and <e> tokens)
    N = len(sentence)
    # The variable p will hold the product
    # that is calculated inside the n-root
    # Update this in the code below
    product_pi = 1.0
    # Index t ranges from n to N - 1, inclusive on both ends
    for t in range(n, N):  # complete this line
        # get the n-gram preceding the word at position t
        n_gram = sentence[t - n:t]
        # get the word at position t
        word = sentence[t]
        probability = ngram.generateProbability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        product_pi *= (1 / probability)
    perplexity = product_pi ** (1 / N)
    return perplexity
