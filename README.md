## Automated-Extractive-Marathi-Text-Summarizer

This project focuses on implementing an extractive text summarization tool for Marathi news articles using the TextRank algorithm. The goal is to automatically generate concise summaries by identifying and extracting the most important sentences from the input text. This tool can be used for processing and summarizing Marathi text, particularly news articles, and can help in reducing the reading time while retaining essential information.

## Overview

The Marathi Text Summarization Tool leverages the TextRank algorithm, a graph-based ranking model, to summarize Marathi news articles in an extractive manner. Extractive summarization involves identifying the most relevant sentences in a text and concatenating them to form a summary, rather than generating entirely new sentences.

This tool is especially designed to handle Marathi language text, which presents unique challenges in terms of tokenization, stemming, and stopword removal due to its morphology and syntax. A user-friendly web interface has been developed using Streamlit, which allows users to interact with the summarizer without needing advanced technical knowledge.

## Technologies Used

This project integrates several powerful tools and libraries to achieve the desired functionality:

1. Python: The main programming language used for developing the text summarization tool and building the application.
2. Streamlit: A fast, easy-to-use Python library to create web apps, used here for creating an intuitive graphical user interface (GUI) for interacting with the tool.
3. NLTK (Natural Language Toolkit): A popular library for working with human language data. It is used here for tasks like sentence tokenization, stopword removal, and text processing.
4. NetworkX: This library is used to create and manipulate the graph structures required by the TextRank algorithm for ranking sentences based on their importance.
5. Pandas: A powerful data manipulation library that is used to handle tabular data, structure datasets, and preprocess text in the project.
6. NumPy: A library that provides support for large, multi-dimensional arrays and matrices, used for efficient numerical computations.
7. Scikit-learn: This machine learning library is used for text vectorization and transforming raw text data into numerical formats for processing.
8. Matplotlib: A plotting library used to visualize data and results, such as showing word frequencies, sentence rankings, and the final summarization.
9. BeautifulSoup: This library is used for web scraping, specifically to extract Marathi news articles from ABP Majha’s website, which forms the dataset for this project.

## Project Structure

    marathi-text-summarization/
    │
    ├── data/               # Contains raw and preprocessed Marathi news articles
    ├── src/                # Source code for text preprocessing and summarization
    ├── app.py              # Main Streamlit application file
    ├── requirements.txt    # List of required Python packages
    ├── README.md           # Project documentation


## Installation

Follow these steps to set up the project on your local machine:

    1. Clone the repository:
        git clone https://github.com/yourusername/marathi-text-summarization.git
        cd marathi-text-summarization

    2. Create and activate a virtual environment:
        python -m venv venv
        source venv/bin/activate  # For Windows: venv\Scripts\activate

    3. Install the required dependencies:
        pip install -r requirements.txt

    4. Run the Streamlit application:
        streamlit run app.py


## Usage

Once the application is up and running, follow these steps to use the summarization tool:

1. Input the Marathi news article: Upload a text file containing the Marathi news article you wish to summarize.
2. Text preprocessing: The system will automatically preprocess the text by performing sentence tokenization, stopword removal, and stemming.
3. Summary generation: The TextRank algorithm will then rank the sentences based on their importance and generate a concise summary.
4. Output: The summarized version of the news article will be displayed on the Streamlit interface.

## Dataset

The dataset used in this project consists of 60 Marathi news articles scraped from the Pune section of the ABP Majha website. Each article contains around 3-4 paragraphs, ranging from 200 to 400 words. The articles are manually cleaned and preprocessed for text analysis.

    Articles Count: 60 news articles.
    Article Length: Each article consists of approximately 3-4 paragraphs.
    Word Count: Articles range from 200 to 400 words each.

## Project Flow

1. Input Text: The user provides a Marathi text file via the Streamlit interface.
2. Text Preprocessing: The input text is tokenized into sentences, and stopwords are removed to focus on    meaningful content. Stemming is applied to reduce words to their root forms.
3. Text Ranking: The TextRank algorithm processes the cleaned text, creating a graph where sentences are nodes, and edges represent similarity between sentences. The algorithm ranks sentences based on their centrality in the graph.
4. Summarization: The highest-ranked sentences are selected to generate the summary. The user can adjust the compression ratio to control the summary length.
5. Output Summary: The summarized text is displayed to the user.

## Future Enhancements

1. Improve Stopword Removal: Refining the list of Marathi stopwords to improve the accuracy of text preprocessing.
2. Add Support for Multiple Articles: Enabling the tool to summarize multiple articles simultaneously.
3. Explore Deep Learning Models: Integrating BERT or other transformer-based models for improved summarization quality.