# -*- coding : utf-8 -*-
# @Time      :2023-04-19 12:44
# @Author   : zy(子永)
# @ Software: Pycharm - windows
# 参考链接 https://github.com/zhudachang1/papercheck
import copy
import os.path

# 自然语言处理包
import jieba
import jieba.analyse
import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import string


class TextSimilarity:
    def __init__(self, checked_doc: str, ori_doc: str, min_len: int = 7, cos_threshold: float = 0.6,
                 language: str = 'zh'):
        # 初始化CountVectorizer
        self.ori_text_list = None
        self.checked_text_list = None
        self.stopwords = None
        self.ori_text = None
        self.checked_text = None
        self.vectorizer = CountVectorizer()
        self.checked_doc = checked_doc
        self.ori_doc = ori_doc
        # 检测最短句子长度
        self.min_len = min_len
        # 余弦相似度阈值
        self.cos_threshold = cos_threshold
        self.language = language
        self.end_symbols = ['。', '！', '？', '；', '……', '…', '，', '\n', '\r', '\t', '.', '!', '?', ';', ',']
        # 结果输出的文件操作指针
        self.fp = None
        self.result_path = None

        # 加载停用词

    def preprocess(self):
        # 去除空白符
        self.checked_text = self.checked_doc.strip().strip('\n').strip('\r').strip('\t')
        self.ori_text = self.ori_doc.strip().strip('\n').strip('\r').strip('\t')
        # 去除标点符号
        self.checked_text = self.checked_doc.translate(str.maketrans('', '', string.punctuation))
        self.ori_text = self.ori_doc.translate(str.maketrans('', '', string.punctuation))

        self.checked_text_list = self.cut_sentence(self.checked_text)
        self.ori_text_list = self.cut_sentence(self.ori_text)
        # print(len(self.checked_text_list))
        # print(len(self.ori_text_list))

    # 读取加载停用词文件
    def load_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.stopwords = set([line.strip() for line in f])
            return [line.strip() for line in f]

    # 根据停用词列表对文本进行过滤，并切成短句
    def cut_sentence(self, text: str) -> list:
        sentences = []
        start = 0
        for i, char in enumerate(text):
            if char in self.end_symbols:
                sentence = text[start:i + 1]
                sentence = sentence.strip().strip('\n').strip('\r').strip('\t').strip('\n')
                if len(sentence) < self.min_len:
                    continue
                sentences.append(sentence)
                start = i + 1
        return copy.deepcopy(sentences)

    @staticmethod
    def get_bow_vectors(sentence1, sentence2, language='zh'):
        # jieba.analyse.set_stop_words(r'./files/stopwords.txt')

        if language == 'zh':
            # 分词
            sentence1_words = jieba.cut(sentence1)
            sentence2_words = jieba.cut(sentence2)

            # 将分词结果转换为字符串列表
            sentence1 = ' '.join(list(sentence1_words))
            sentence2 = ' '.join(list(sentence2_words))

        vectorizer = CountVectorizer()
        try:
            # 将分词后的句子列表转换为词袋向量
            bow_vectors = vectorizer.fit_transform([sentence1, sentence2]).toarray()
            return bow_vectors[0], bow_vectors[1]
        except Exception as e:
            # 返回词向量
            return [1, 0], [0, 1]

    # 初始输出结果的文件指针
    def init_fp(self, path='./result.txt'):
        self.fp = open(path, 'w', encoding='utf-8')
        self.result_path = os.path.abspath(path)
        return self.result_path

    # 对象销毁的时候关闭文件指针
    def __del__(self):
        if self.fp:
            self.fp.close()

    def check(self):
        self.preprocess()
        self.init_fp()
        for checked_sentence in tqdm.tqdm(self.checked_text_list):
            for ori_sentence in self.ori_text_list:
                # temp_similarity = self.cos_distance(checked_sentence, ori_sentence)
                temp_similarity = self.jaro_winkler_distance(checked_sentence, ori_sentence)

                if temp_similarity > self.cos_threshold:
                    info = f'\n {temp_similarity * 100}% \nchecked_sentence: ' \
                           f'{checked_sentence}\nori_sentence: \t{ori_sentence}\n------------- '
                    self.fp.write(info)
                    # print(info)
        self.fp.close()

    def cos_distance(self, checked_sentence, ori_sentence):
        temp_vectors = self.get_bow_vectors(checked_sentence, ori_sentence, language=self.language)
        return self.cosine_similarity(temp_vectors[0], temp_vectors[1])

    @staticmethod
    def sift4_distance(checked_sentence, ori_sentence):
        from strsimpy.sift4 import SIFT4
        sift4 = SIFT4()
        sim = sift4.distance(checked_sentence, ori_sentence)
        return sim

    @staticmethod
    def jaro_winkler_distance(checked_sentence, ori_sentence):
        # print(checked_sentence, ori_sentence)
        from strsimpy.jaro_winkler import JaroWinkler
        jw = JaroWinkler()
        sim = jw.similarity(checked_sentence, ori_sentence)
        return sim

    @staticmethod
    def cosine_similarity(vec1, vec2):
        # 计算两个向量的余弦距离
        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        # 除零处理
        if denominator == 0:
            return 0
        return numerator / denominator
