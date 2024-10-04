import streamlit as st
import collections
import math
import networkx as nx
import operator
from textpre import cleanText
from wl import process_word_list

def calculate_word_scores(word_dict, size):
    window = 3
    nodeHash = {}
    word_score = collections.defaultdict(dict)
    count = 0

    for i in word_dict.keys():
        for j in range(0, len(word_dict[i])):
            count += 1
            position = float(count) / (float(size) + 1.0)
            word_score[i][j] = 1.0 / (math.pi * math.sqrt(position * (1 - position)))
            word = word_dict[i][j]
            if word in nodeHash:
                if nodeHash[word] < word_score[i][j]:
                    nodeHash[word] = word_score[i][j]
            else:
                nodeHash[word] = word_score[i][j]

    return word_score, nodeHash

def build_graph(word_dict, nodeHash, window):
    graph = nx.Graph()
    graph.add_nodes_from(nodeHash.keys())

    for i in word_dict.keys():
        for j in range(0, len(word_dict[i])):
            current_word = word_dict[i][j]
            next_words = word_dict[i][j + 1:j + window]
            for word in next_words:
                graph.add_edge(current_word, word, weight=(nodeHash[current_word] + nodeHash[word]) / 2)

    return graph

def text_rank_summary(sentences, word_dict, textRank, keyphrases, numberofSentences):
    sentenceScore = {}

    for i in word_dict.keys():
        position = float(i + 1) / (float(len(sentences)) + 1.0)
        positionalFeatureWeight = 1.0 / (math.pi * math.sqrt(position * (1.0 - position)))
        sumKeyPhrases = 0.0

        for keyphrase in keyphrases:
            if keyphrase in word_dict[i]:
                sumKeyPhrases += textRank[keyphrase]

        sentenceScore[i] = sumKeyPhrases * positionalFeatureWeight

    sSentenceScores = sorted(sentenceScore.items(), key=operator.itemgetter(1), reverse=True)[:numberofSentences]
    sSentenceScores = sorted(sSentenceScores, key=operator.itemgetter(0), reverse=False)

    summary = [sentences[s[0]] for s in sSentenceScores]
    return summary

st.title("Marathi Text Summarization")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    with open("uploaded_text.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    word_dict, sentences, size = cleanText("uploaded_text.txt")
    word_score, nodeHash = calculate_word_scores(word_dict, size)
    graph = build_graph(word_dict, nodeHash, window=3)
    textRank = nx.pagerank(graph, weight='weight')

    n = int(math.ceil(min(0.1 * size, 7 * math.log(size))))
    keyphrases = sorted(textRank, key=textRank.get, reverse=True)[:n]

    summary = text_rank_summary(sentences, word_dict, textRank, keyphrases, numberofSentences=6)

    st.write("## Summary")
    for sentence in summary:
        st.write(sentence)

    # st.write("## Key Phrases")
    # for phrase in keyphrases:
    #     st.write(phrase)
