# -*- coding: UTF-8 -*-
"""
    爬取博客园博客，并保存txt文件
"""

import requests
import pandas as pd

from lxml import html

def sub_page(url, headers):
    content = requests.get(url, headers).content.decode('utf-8')
    html = etree.HTML(content)

    title = html.xpath('//a[@id="cb_post_title_url"]/text()')
    text = html.xpath('string(//*[@id="cnblogs_post_body"])')
    return pd.Series([title, text], index=["title", "text"])

def main_page(url, headers):
    content = requests.get(url, headers).content.decode('utf-8')
    html = etree.HTML(content)

    urls = html.xpath('//div[@class="post_item_body"]/h3/a/@href')
    for u in urls:
        data = sub_page(u, headers)
        print(data)

def pagination(domain, src, headers):
    url = domain + src
    while True:
        main_page(url, headers)
        content = requests.get(url, headers).content.decode('utf-8')
        html = etree.HTML(content)

        next = html.xpath('//div[@class="pager"]/a[last()]')
        if next[0].xpath('text()')[0] == 'Next >':
            url = domain + next[0].xpath('@href')[0]
            print(url)
        else:
            break

if __name__ == "__main__":
    domain = "https://www.cnblogs.com"
    src = "/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"
    }
    etree = html.etree
    pagination(domain, src, headers)
