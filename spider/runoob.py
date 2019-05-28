# -*- coding: UTF-8 -*-
"""
    爬取http://www.runoob.com中python100例，并保存csv文件
"""

import requests
import pandas as pd

from bs4 import BeautifulSoup


# 子页面
def sub_page(url, headers):
    content = requests.get(url, headers).content.decode('utf-8')
    html = BeautifulSoup(content, 'lxml')
    div_tag = html.find(id='content')

    title = div_tag.h1.text
    p_tag = div_tag.find_all('p')
    topic = p_tag[1].text

    analysis = p_tag[2].text
    source_tag = div_tag.find('div', class_='example_code')
    source = ""
    if source_tag:
        source = source_tag.text
    return pd.Series([title, topic, analysis, source], index=['title', 'topic', 'analysis', 'source'])


# 主页面
def main_page(url, headers):
    content = requests.get(url, headers).content.decode('utf-8')
    html = BeautifulSoup(content, 'lxml')
    a_tag = html.find(id='content').ul.find_all(name='a')

    list = []
    for a in a_tag:
        list.append(a.attrs["href"])

    return list


if __name__ == '__main__':
    domain = "http://www.runoob.com"
    src = "/python/python-100-examples.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"
    }

    url = main_page(domain + src, headers)
    df = pd.DataFrame(columns=['title', 'topic', 'analysis', 'source'])
    count = 1
    for u in url:
        data = sub_page(domain + u, headers)
        df = df.append(data, ignore_index=True)

        print(str(count) + " done")
        count += 1
    df.to_csv("runoob.csv", index=True, header=True, encoding="utf_8_sig")

