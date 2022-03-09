import random
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from preprocessing import clean_text, expand_contractions, remove_non_alphabetical_character, remove_single_letter, remove_stopwords, lemmatize_text

df = pickle.load(open('dataset.sav', 'rb'))
title = df['Title']
body = df['Body']
top_tf_idf_title = df['Cleaned_Title_top_words_vector']
top_tf_idf_body = df['Cleaned_Body_top_words_vector']
index_map = pickle.load(open('inverted_index.sav', 'rb'))
documents = []
tf_idf_body_vectorizer = pickle.load(open('tf_idf_body_vectorizer.sav','rb'))
tf_idf_title_vectorizer = pickle.load(open('tf_idf_title_vectorizer.sav','rb'))


class Document:
    def __init__(self, title, text, tf_idf_title, tf_idf_body):
        self.title = title
        self.text = text
        self.tf_idf_title = tf_idf_title
        self.tf_idf_body = tf_idf_body
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']

    def get_tf_idf_title(self, tf_idf_query_title):
        cosine_similarities = cosine_similarity(self.tf_idf_title[0].reshape(1, -1), tf_idf_query_title.reshape(1, -1))
        return cosine_similarities[0][0]

    def get_tf_idf_body(self, tf_idf_query_body):
        cosine_similarities = cosine_similarity(self.tf_idf_body[0].reshape(1, -1), tf_idf_query_body.reshape(1, -1))
        return cosine_similarities[0][0]


def build_index():
    # добавляем все наши документы
    df = pickle.load(open('dataset.sav', 'rb'))
    index_map = pickle.load(open('inverted_index.sav', 'rb'))
    top_tf_idf_title = df['Cleaned_Title_top_words_vector'].tolist()
    top_tf_idf_body = df['Cleaned_Body_top_words_vector'].tolist()
    title = df['Title'].tolist()
    body = df['Body'].tolist()


def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    query_title = tf_idf_title_vectorizer.transform([query])
    query_body = tf_idf_body_vectorizer.transform([query])
    body_score = document.get_tf_idf_body(query_body)
    title_score = document.get_tf_idf_title(query_title)
    return 0.2 * body_score + 0.8 * title_score


def retrieve(query):
    # возвращает начальный список релевантных документов
    candidates = []
    if query == '':
        for idx in range(50):
            candidates.append(Document(title[idx], body[idx], top_tf_idf_title[idx], top_tf_idf_body[idx]))
        return candidates[:50]
    # чистим запрос под наш инвертированный индекс
    cleaned_query = clean_text(query)
    cleaned_query = cleaned_query.lower()
    cleaned_query = expand_contractions(cleaned_query)
    cleaned_query = remove_non_alphabetical_character(cleaned_query)
    cleaned_query = remove_single_letter(cleaned_query)
    cleaned_query = remove_stopwords(cleaned_query)
    cleaned_query = lemmatize_text(cleaned_query)

    keywords = cleaned_query.split()
    list_id = index_map[keywords[0]]

    # объединяю списки индексов(два указателя)
    for word in keywords[1:]:
        i = j = 0
        time_list = []
        while i < len(list_id) and j < len(index_map[word]):
            if list_id[i] == index_map[word][j]:
                time_list.append(list_id[i])
            if list_id[i] < index_map[word][j]:
                i += 1
            else:
                j += 1
        list_id = time_list

    # возвращаю подходящие документы
    for idx in list_id:
        candidates.append(Document(title[idx], body[idx], top_tf_idf_title[idx], top_tf_idf_body[idx]))

    return candidates[:50]
