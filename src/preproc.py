import nltk
from src.constants import  DATASET_FILE
nltk.data.path.append('.')

def load_data():
    """
    This function loads the data from the text file by opening it and reading its contents.
    ***
    Args:
    N/A
    ***
    Returns:
    data: String of data
    """
    with open(DATASET_FILE, "r", encoding='utf-8') as f:
        data = f.read()
    return data


def splitSentences(sentences):
    """
    This function splits the string data on newline character into a list of sentences.
    ***
    Args:
        sentences: str
    ***
    Returns:
        sentences_list: A list of sentences
    """
    sentences_list = sentences.split('\n')
    # - Remove whitespaces from the beginning and end of each sentence
    sentences_list = [s.strip() for s in sentences_list]
    # Remove sentences with length 0
    sentences_list = [s for s in sentences_list if len(s) > 0]
    return sentences_list

def tokenizeSentences(sentences_list):
    """
    This function splits the sentences into words or tokens.
    ***
    Args:
        sentences: List of strings
    ***
    Returns:
       tokenized_sentences: List of lists of words or tokens
    """
    tokenized_sentences = []
    for sentence in sentences_list:
        # Change the sentence to lowercase
        sentence = sentence.lower()
        # Use nltk's word tokenizer to split the senteces into tokens/words
        tokenized =  nltk.word_tokenize(sentence)
        # Add the token to a list
        tokenized_sentences.append(tokenized)
    return tokenized_sentences
