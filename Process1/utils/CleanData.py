import pandas as pd
import re

cnt = 0


def deal_text(txt):
    htmlpattern = r"&[^ ]+?;"
    txt = re.sub(htmlpattern, "", txt)

    words = txt.split("; Title")
    global cnt
    if len(words) != 2:
        cnt += 1
        print("============%d" % cnt)
        for w in words:
            print(w)
    else:
        txt = "Title is" + words[1] + "; " + words[0]

    # 去除描述开头
    txt = txt.replace("Description: ", "")

    words = txt.split()
    text = ' '.join(words)
    text = text.lower()
    follow_pattern = r"(you can )?follow his adventures.*?\."
    learn_more_pattern = r"(you can |to )?learn more( about| at|,|:) .*?\.(com|net)"
    find_out_pattern = r"(you can )?find out more .*?\.(com|net|uk)"
    find_pattern = r"(you can )?find (him|her) .*?\.(com|net|uk)"
    visit_pattern = r"(you can )?visit (him|her) .*?\.(com|net|uk)"
    text = re.sub(follow_pattern, '', text)
    text = re.sub(learn_more_pattern, '', text)
    text = re.sub(find_pattern, '', text)
    text = re.sub(find_out_pattern, '', text)
    text = re.sub(visit_pattern, '', text)
    pattern = r'(\.html|\.net|\.com)([A-Z])'
    text = re.sub(pattern, r'\1 \2', text)

    text = text.replace("http//", "http://")
    text = text.replace("http;//", "http://")
    text = text.replace("http: //", "http://")
    text = text.replace("https://", "https://")
    text = text.replace("www. ", "www")
    text = text.replace(" ()", "")
    text = text.replace("()", "")
    text = text.replace("<br>", "")
    text = text.replace(".com", "")
    text = text.replace("//: ", "")

    copyrightPattern1 = r"(copyright|Copyright).*?inc[\.]?"
    copyrightPattern2 = r"(copyright|Copyright).*?[\.]"
    text = re.sub(copyrightPattern1, "", text)
    len1 = len(text)
    text = re.sub(copyrightPattern2, "", text)
    if (len1 - len(text) > 200):
        cnt += 1
        print("-----------%d" % cnt)
        print(len1 - len(text))
        print(text)
    text = text.replace("all rights reserved", '')

    webPattern = r"(https?:\/\/)?(([0-9a-z.]+\.[a-z]+)|(([0-9]{1,3}\.){3}[0-9]{1,3}))(:[0-9]+)?(\/[0-9a-z%/.\-_]*)?(\?[0-9a-z=&%_\-]*)?(\#[0-9a-z=&%_\-]*)?"
    text = re.sub(webPattern, "", text)



    text = text.replace("   ", " ")
    return text


def load_data_rf():
    # 读取数据
    train = pd.read_csv("../Data/train.csv")
    test = pd.read_csv("../Data/test.csv")

    # 切换文字位置
    # train["text"] = train["text"].apply(lambda x: x.split(";")[1] + x.split(";")[0])
    # test["text"] = test["text"].apply(lambda x: x.split(";")[1] + x.split(";")[0])

    #
    # # 去除标题开头
    # train["text"] = train["text"].apply(lambda x: x.replace("Title: ", ""))
    # test["text"] = test["text"].apply(lambda x: x.replace("Title: ", ""))

    train["text"] = train["text"].apply(lambda x: deal_text(x))
    test["text"] = test["text"].apply(lambda x: deal_text(x))

    train.to_csv('../Data/train_new.csv')
    test.to_csv('../Data/test_new.csv')


if __name__ == '__main__':
    load_data_rf()
