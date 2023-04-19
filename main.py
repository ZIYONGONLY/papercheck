# -*- coding : utf-8 -*-
# @Time      :2023-04-19 12:34
# @Author   : zy(子永)
# @ Software: Pycharm - windows

from check.cos import TextSimilarity

from check.read_file import read_docx_file, read_pdf_file, read_txt_file

# 修改为自己的文件路径，origin_file_path为原文，checked_file_path为待检测的文本
origin_file_path = r"E:\tmp\checked.docx"
checked_file_path = r"E:\tmp\origin.docx"


def main():
    origin_file = read_docx_file(origin_file_path)
    checked_file = read_docx_file(checked_file_path)
    ts = TextSimilarity(checked_file, origin_file, min_len=10, cos_threshold=0.9, language='zh')
    ts.check()


# 测试
if __name__ == '__main__':
    main()
