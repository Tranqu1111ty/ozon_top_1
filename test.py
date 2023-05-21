# from tensorflow.keras.preprocessing.text import Tokenizer
#
#
# def encode_sentences(sentences):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(sentences)
#     encoded_sentences = tokenizer.texts_to_sequences(sentences)
#     return encoded_sentences
#
#
# sentences = [
#     "Это первое предложение.",
#     "Это второе предложение.",
#     "И это третье предложение."
# ]
# encoded_sentences = encode_sentences(sentences)
#
# print("Исходные предложения:", sentences)
# print("Закодированные предложения:", encoded_sentences)


# from gensim.models import Word2Vec
# from nltk.tokenize import CharTokenizer
#
# def train_char_embeddings(sentences, embedding_dim=100, window=5, min_count=1):
#     # Токенизация предложений на уровне символов
#     tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
#
#     # Обучение модели Word2Vec на уровне символов
#     model = Word2Vec(tokenized_sentences, size=embedding_dim, window=window, min_count=min_count)
#
#     return model
#
# def get_char_embeddings(sentence, model):
#     # Токенизация предложения на уровне символов
#     tokenized_sentence = tokenizer.tokenize(sentence)
#
#     # Получение эмбеддингов для каждого символа в предложении
#     embeddings = [model.wv[char] for char in tokenized_sentence]
#
#     return embeddings
#
# # Пример использования функций
# sentences = [
#     "Это первое предложение.",
#     "Это второе предложение.",
#     "И это третье предложение."
# ]
#
# tokenizer = CharTokenizer()
#
# # Обучение модели эмбеддингов на уровне символов
# model = train_char_embeddings(sentences)
#
# # Получение эмбеддингов для предложения
# sentence = "Это первое предложение."
# embeddings = get_char_embeddings(sentence, model)
#
# print("Исходное предложение:", sentence)
# print("Эмбеддинги:", embeddings)


# from sklearn.feature_extraction.text import CountVectorizer
#
# def encode_sentences(sentences):
#     vectorizer = CountVectorizer()
#     encoded_sentences = vectorizer.fit_transform(sentences)
#     return encoded_sentences.toarray()
#
# sentences = [
#     "Это первое предложение.",
#     "Это второе предложение.",
#     "И это третье предложение."
# ]
# encoded_sentences = encode_sentences(sentences)
#
# print("Исходные предложения:", sentences)
# print("Закодированные предложения:")
# for encoded_sentence in encoded_sentences:
#     print(encoded_sentence)

from sklearn.feature_extraction.text import CountVectorizer


def create_bag_of_words(sentences):
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(sentences)
    len_sentences = []
    for sentence in sentences:
        len_sentences.append(len(sentence.split()))

    return bag_of_words.toarray(), len_sentences


sentences = [
    "Первое предложение ",
    "Это типо второе предложение",
    "Ну а это пусть будет третье предложение"
]
bag_of_words, len_sentences = create_bag_of_words(sentences)
print(bag_of_words)
print("Мешок слов:")
for i, sentence in enumerate(sentences):
    print(sentence)
    print(bag_of_words[i], len_sentences[i])
    print()

