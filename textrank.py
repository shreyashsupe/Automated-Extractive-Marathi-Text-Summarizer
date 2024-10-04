# import collections
# import math
# import operator
# import networkx as nx
# import re
# from nltk.tokenize import sent_tokenize

# # Define Marathi stopwords
# MARATHI_STOPWORDS = {
#     "अधिक", "अनेक", "अशी", "असलयाचे", "असलेल्या", "असा", "असून", "असे", "आज", "आणि", "आता", "आपल्या",
#     "आला", "आली", "आले", "आहे", "आहेत", "एक", "एका", "कमी", "करणयात", "करून", "का", "काम", "काय", "काही",
#     "किवा", "की", "केला", "केली", "केले", "कोटी", "गेल्या", "घेऊन", "झाला", "झाली", "झाले", "झालेल्या",
#     "टा", "डॉ", "तर", "तरी", "तसेच", "ता", "ती", "तीन", "ते", "तो", "त्या", "त्याचा", "त्याची", "त्याच्या",
#     "त्याना", "त्यानी", "त्यामुळे", "त्री", "दिली", "दोन", "न", "नाही", "निर्ण्य", "पण", "पम", "परयतन",
#     "म", "मात्र", "माहिती", "मी", "मुबी", "म्हणजे", "म्हणाले", "म्हणून", "या", "याचा", "याची", "याच्या",
#     "याना", "यानी", "येणार", "येत", "येथील", "येथे", "लाख", "व", "व्यकत", "सर्व", "सागित्ले", "सुरू",
#     "हजार", "हा", "ही", "हे", "होणार", "होत", "होता", "होती", "होते"
# }

# def text_rank_summary(text):
#     nodeHash = {}
#     textRank = {}
#     word_dict = collections.defaultdict(dict)
#     size = 0
#     sentences = []

#     def clean_text(text):
#         text = text.lower()
#         text = re.sub(r"[।]", " ", text)  # Replace Marathi full stop with space
#         text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
#         return text

#     def tokenize_sentences(text):
#         return sent_tokenize(text)

#     def calculate_word_scores(word_dict, size):
#         word_score = collections.defaultdict(dict)
#         count = 0
#         for i in word_dict.keys():
#             for j in range(0, len(word_dict[i])):
#                 count += 1
#                 position = float(count) / (float(size) + 1.0)
#                 word_score[i][j] = 1.0 / (math.pi * math.sqrt(position * (1 - position)))
#                 word = word_dict[i][j]
#                 if word in nodeHash:
#                     if nodeHash[word] < word_score[i][j]:
#                         nodeHash[word] = word_score[i][j]
#                 else:
#                     nodeHash[word] = word_score[i][j]
#         return nodeHash

#     def build_graph(word_dict, nodeHash):
#         graph = nx.Graph()
#         graph.add_nodes_from(nodeHash.keys())
#         window = 3
#         for i in word_dict.keys():
#             for j in range(0, len(word_dict[i])):
#                 current_word = word_dict[i][j]
#                 next_words = word_dict[i][j + 1:j + window]
#                 for word in next_words:
#                     graph.add_edge(current_word, word, weight=(nodeHash[current_word] + nodeHash[word]) / 2)
#         return graph

#     def calculate_sentence_scores(word_dict, sentences, nodeHash):
#         sentenceScore = {}
#         n = int(math.ceil(min(0.1 * len(sentences), 7 * math.log(len(sentences)))))
#         for i in word_dict.keys():
#             position = float(i + 1) / (float(len(sentences)) + 1.0)
#             positionalFeatureWeight = 1.0 / (math.pi * math.sqrt(position * (1.0 - position)))
#             sumKeyPhrases = 0.0
#             for j, sentence in enumerate(sentences):
#                 if any(word in sentence for word in word_dict[i]):
#                     sumKeyPhrases += textRank[j]
#             sentenceScore[i] = sumKeyPhrases * positionalFeatureWeight
#         sSentenceScores = sorted(sentenceScore.items(), key=operator.itemgetter(1), reverse=True)[:n]
#         return sSentenceScores

#     def generate_summary(sSentenceScores, sentences):
#         summary = ""
#         for i in range(0, len(sSentenceScores)):
#             summary += sentences[sSentenceScores[i][0]] + "\n"
#         return summary

#     text = clean_text(text)
#     sentences = tokenize_sentences(text)

#     for i, sentence in enumerate(sentences):
#         words = sentence.split()
#         for j, word in enumerate(words):
#             if word not in MARATHI_STOPWORDS:
#                 word_dict[i][j] = word

#     size = len(sentences)
#     nodeHash = calculate_word_scores(word_dict, size)
#     graph = build_graph(word_dict, nodeHash)
#     textRank = nx.pagerank(graph, weight='weight')
#     sSentenceScores = calculate_sentence_scores(word_dict, sentences, textRank)
#     summary = generate_summary(sSentenceScores, sentences)

#     return summary

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import io

def calculate_sentence_scores(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    X = normalize(X)
    similarity_matrix = cosine_similarity(X)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    return scores

def text_rank_summary(text):
    sentences = text.split("\n")
    sentence_scores = calculate_sentence_scores(sentences)
    ranked_sentences = sorted(((sentence_scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    summary = " ".join([sentence for score, sentence in ranked_sentences[:3]])  # Get the top 3 sentences as summary
    return summary

