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

"""
# 1 读取文件 获得字符串 截取绪论到参考文献部分
# 2 分句 （按句子截止符划分/按设定长度划分【粗分/细分】）
# 3 去除空白内容
# 4 生成句子列表
# 5 去除停用词 （可选）
# 6 计算句子相似度

"""


class TextSimilarity:
    def __init__(self, checked_doc: str, ori_doc: str, min_len: int = 7, cos_threshold: float = 0.6,
                 language: str = 'zh', begin_symbols_intro: str = "绪论", end_symbols_ref: str = "参考文献"):
        self.ori_text_list = None
        self.checked_text_list = None
        self.stopwords = None
        self.ori_text = None
        self.checked_text = None
        self.vectorizer = CountVectorizer()
        self.checked_doc = checked_doc
        self.ori_doc = ori_doc
        self.begin_symbols_intro = begin_symbols_intro
        self.end_symbols_ref = end_symbols_ref
        # 检测最短句子长度
        self.min_len = min_len
        # 余弦相似度阈值
        self.cos_threshold = cos_threshold
        self.language = language
        self.end_symbols = ['。', '！', '？', '；', '……', '…', '，', '\n', '\r', '\t', '.', '!', '?', ';', ',']
        # 结果输出的文件操作指针
        self.fp = None
        self.result_path = None

    # 删除无关的内容部分
    def delete_unrelated_content(self):
        self.checked_text = self._delete_unrelated_content(self.checked_text)
        self.ori_text = self._delete_unrelated_content(self.ori_text)

    # 删除无关的内容部分
    def _delete_unrelated_content(self, text: str):
        # 从头开始找
        begin_index = text.find(self.begin_symbols_intro)
        # 从尾部开始找
        end_index = text.rfind(self.end_symbols_ref)
        # 如果找到了
        if begin_index != -1 and end_index != -1:
            # 截取
            text = text[begin_index:end_index]
            return text
        else:
            return text

    def preprocess(self):
        # 去除空白符
        self.checked_text = self.checked_doc.strip().strip('\n').strip('\r').strip('\t')
        self.ori_text = self.ori_doc.strip().strip('\n').strip('\r').strip('\t')

        # 删除无关的内容部分
        self.delete_unrelated_content()

        self.checked_text_list = self.cut_sentence(self.checked_text)
        self.ori_text_list = self.cut_sentence(self.ori_text)

    # 读取加载停用词文件
    def load_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.stopwords = set([line.strip() for line in f])
            return [line.strip() for line in f]

    # 根据停用词列表对文本进行过滤，并切成短句
    def cut_sentence(self, text: str) -> list:
        # print()
        sentences = []
        start = 0
        for i, char in enumerate(text):
            if char in self.end_symbols:
                sentence = text[start:i]
                sentence = self.remove_blank(sentence)
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
        self.fp.write('检测结果：\n')
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
                    info = f'\n {temp_similarity * 100}% \n* checked_sentence: ' \
                           f'{checked_sentence}\n* ori_sentence: \t{ori_sentence}\n------------- '
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

    # 仅考虑连续重复的字数进行重复检测
    def check_repeat(self):
        self.preprocess()
        self.init_fp()
        for checked_sentence in tqdm.tqdm(self.checked_text_list):
            for ori_sentence in self.ori_text_list:
                temp_similarity = self.repeat_check(checked_sentence, ori_sentence)
                if temp_similarity > self.cos_threshold:
                    info = f'\n {temp_similarity * 100}% \n* checked_sentence: ' \
                           f'{checked_sentence}\n* ori_sentence: \t{ori_sentence}\n------------- '
                    self.fp.write(info)
                    # print(info)
        self.fp.close()

    def repeat_check(self, checked_sentence, ori_sentence):
        checked_sentence = checked_sentence.replace(' ', '')
        ori_sentence = ori_sentence.replace(' ', '')
        temp_similarity = 0
        for i in range(len(checked_sentence)):
            for j in range(len(ori_sentence)):
                if checked_sentence[i] == ori_sentence[j]:
                    temp_similarity += 1
                    break
        temp_similarity /= len(checked_sentence)
        return temp_similarity

    # 去除字符串中所有空白内容
    @staticmethod
    def remove_blank(text):
        return text.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '').replace('　', '')
