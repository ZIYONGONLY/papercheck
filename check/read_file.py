# -*- coding : utf-8 -*-
# @Time      :2023-04-19 16:30
# @Author   : zy(子永)
# @ Software: Pycharm - windows
# 读取文件，获取文件内容
import docx
import pdfplumber


# 读取docx文件
def read_docx_file(file_path):
    doc = docx.Document(file_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


# 读取txt文件
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# 读取PDF文件

def read_pdf_file(file_path):
    with pdfplumber.open(file_path) as pdf:
        all_text = ''
        for page in pdf.pages:
            text = page.extract_text()  # 提取文本
            # print(text)
            all_text += text
        return all_text
