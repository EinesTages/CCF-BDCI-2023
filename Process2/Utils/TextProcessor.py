import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def clean_text(text):
    stemmer = SnowballStemmer("english")
    # 将文本转换成小写字写,替换缩写
    text = text.lower().replace("it's", "it is").replace("i'm", "i am").replace("he's", "he is").replace("she's",
                                                                                                         "she is") \
        .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's",
                                                                                                        "that is") \
        .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't",
                                                                                                     "does not") \
        .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not") \
        .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's",
                                                                                                          "let us")
    text = re.sub("[^a-zA-Z]", " ", text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]  # 去除停用词的同时使用词干还原
    text = ' '.join(words)
    return text


def tokenizer(reviews):
    Words = []
    for review in reviews:
        Words.append(clean_text(review[12:]))
    return Words
