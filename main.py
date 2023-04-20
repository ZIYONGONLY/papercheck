# -*- coding : utf-8 -*-
# @Time      :2023-04-19 12:34
# @Author   : zy(子永)
# @ Software: Pycharm - windows
import os
from check.cos import TextSimilarity
import gradio as gr
from check.read_file import read_docx_file, read_pdf_file, read_txt_file

# 修改为自己的文件路径，origin_file_path为原文，checked_file_path为待检测的文本
result_path = None


def main(origin_file_path, checked_file_path, min_len=10, cos_threshold=0.9):
    try:
        print(origin_file_path.name, checked_file_path.name)
        # 判断文件类型
        if origin_file_path.name.endswith('.docx'):
            origin_file = read_docx_file(origin_file_path.name)
        elif origin_file_path.name.endswith('.txt'):
            origin_file = read_txt_file(origin_file_path.name)
        elif origin_file_path.name.endswith('.pdf'):
            origin_file = read_pdf_file(origin_file_path.name)
        else:
            return '原文文件格式不支持'
        # 判断文件类型
        if checked_file_path.name.endswith('.docx'):
            checked_file = read_docx_file(checked_file_path.name)
        elif checked_file_path.name.endswith('.txt'):
            checked_file = read_txt_file(checked_file_path.name)
        elif checked_file_path.name.endswith('.pdf'):
            checked_file = read_pdf_file(checked_file_path.name)
        else:
            return '待检测文本文件格式不支持'
        # print(origin_file, checked_file)
        ts = TextSimilarity(checked_file, origin_file, min_len=min_len, cos_threshold=cos_threshold, language='zh')
        ts.check()
        global result_path
        result_path = ts.result_path
        return '检测结果保存在 ' + ts.result_path
    except Exception as e:
        return str(e)


# 打开检测结果
def open_result():
    from subprocess import run
    if result_path:
        run(f'notepad {result_path}', shell=True)
    else:
        return '文件不存在'


def load_result():
    global result_path
    if result_path:
        with open(result_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
            full_text = full_text.replace('\n', '<br>')
            return full_text
    else:
        return '文件不存在'


# 测试
if __name__ == '__main__':
    with gr.Blocks() as app:
        with gr.Tabs():
            with gr.Row():
                origin_file = gr.inputs.File(label="原文 仅支持docx/ PDF/ txt")
                checked_file = gr.inputs.File(label="待检测文本 仅支持docx/ PDF/ txt")
            with gr.Row():
                min_len = gr.Slider(minimum=0, maximum=100, default=10, step=1, value=10, label="忽略的最小长度")
                cos_threshold = gr.Slider(minimum=0, maximum=1, default=0.9, step=0.01, value=0.9, label="相似度阈值")
                check_button = gr.Button("检测")
                result_button = gr.Button("加载检测结果")
                result_click_button = gr.Button("打开检测结果")
            with gr.Row():
                score = gr.Label(label="output")
        check_button.click(fn=main, inputs=[origin_file, checked_file, min_len, cos_threshold], outputs=score)
        result_button.click(fn=load_result, outputs=score)
        result_click_button.click(fn=open_result)
    app.launch()
