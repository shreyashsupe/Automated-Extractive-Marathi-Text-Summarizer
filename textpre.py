import re
import collections

def cleanText(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = text.split('.')
    word_dict = collections.defaultdict(list)
    size = 0

    for i, sentence in enumerate(sentences):
        words = re.findall(r'\b\w+\b', sentence.lower())
        word_dict[i] = words
        size += len(words)

    return word_dict, sentences, size
