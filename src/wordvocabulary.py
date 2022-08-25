from src.constants import UNKNOWN_TOKEN


def countWords(tokenized_sentences):
    """
    This function Count the number of occurrence of each word in the list.
    ***
    Args:
    tokenized_sentences: List of lists of words/tokens of sentences
    ***
    Returns:
    word_counts: dict containing word as key and its frequency as value
    """
    word_counts = {}
    for sentence in tokenized_sentences:
        for word in sentence:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
    return word_counts


def getHighFreqWords(tokenized_sentences, count_threshold):
    """
    This functions finds a list of words that have a higher frequency than count threshold
    ***
    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: int value used as threshold for high frequency words.
    ***
    Returns:
        high_freq_words: List of words that appear more than count_threshold number of times.
    """
    high_freq_words = []
    # Get the frequency of words in input
    word_counts = countWords(tokenized_sentences)
    for word, count in word_counts.items():
        if count >= count_threshold:
            high_freq_words.append(word)
    return high_freq_words


def addUnkownToken(tokenized_sentences, vocabulary, unknown_token=UNKNOWN_TOKEN):
    """
    This function adds the unknown token in place of low frequency words
    ***
    Args:
        tokenized_sentences: List of lists of strings
        high_freq_words: list of words with high frequency
        unknown_token: Token to represent unknown words
    ***
    Returns:
        processed_tokenized_sentences: List of lists of tokens with <UNK> for low frequency words
    """
    high_freq_words = set(vocabulary)
    processed_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in high_freq_words:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        # Add the list of tokens to the list of lists
        processed_tokenized_sentences.append(replaced_sentence)
    return processed_tokenized_sentences
