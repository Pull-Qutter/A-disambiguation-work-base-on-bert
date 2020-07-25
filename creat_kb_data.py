import re
import os
from urllib import request
from urllib.parse import quote
from bs4 import BeautifulSoup as sp
import pickle
import json
from collections import defaultdict

'''
1.爬虫获取知识库实体描述kbdata_Net
2.根据train补充爬虫没获取到的实体描述,构建知识库kb_data
'''

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
bracket = re.compile(r'\[\d*]')
Max_len = 512


def Get_Entity_Description(entry):
    url = "https://baike.baidu.com/item/" + quote(entry)
    req = request.Request(url, headers=header)
    html = request.urlopen(req).read()
    soup = sp(html, "html.parser")
    Description_len = 10
    content = soup.findAll('div', {'class': 'para'})
    if content:
        for idx, i in enumerate(content):
            i = i.get_text()
            i = i.replace('\n', '')
            i = i.replace('\r', '')
            i = i.replace(u'\xa0', u'');
            i = re.sub(bracket, '', i)
            if (len(i) > Description_len):
                break
        return i if len(i) < Max_len else i[:Max_len]
    else:
        return 'NIL'


# 生成爬虫知识库
if os.path.exists('./data_deal/kb_data/kb_dataNet.pkl'):
    with open('./data_deal/kb_data/kb_dataNet.pkl', 'rb') as f:
        kb_dataNet = pickle.load(f)
else:
    kb_dataNet = defaultdict(dict)
    f = open('./data/company_2_code_sub.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        [kb_id, stock_name, stock_full_name, stock_code] = line.split()
        temp_data = Get_Entity_Description(stock_full_name)
        kb_dataNet.update({kb_id: {
            'kb_id': kb_id,
            'mention': stock_name,
            'stock_full_name': stock_full_name,
            'data': temp_data
        }})
        print('%d:%d' % (idx - 1, len(kb_dataNet[kb_id]['data'])))
        print(kb_dataNet[kb_id])
        # 保存好爬虫到的kb_dataNet
        with open('./data_deal/kb_data/kb_dataNet.pkl', 'wb') as f:
            pickle.dump(kb_dataNet, f)

        # with open('./data_deal/kb_data/kb_dataNet.json', 'w',encoding='utf-8') as f:
        #     for each_dict in kb_dataNet:
        #         f.write(json.dumps(each_dict,ensure_ascii=False)+'\n')

# 利用训练集生成本地知识库
if os.path.exists('./data_deal/kb_data/kb_dataLoc.pkl'):
    with open('./data_deal/kb_data/kb_dataLoc.pkl', 'rb') as f:
        kb_dataLoc = pickle.load(f)
else:
    f = open('./data/train.json', 'r', encoding='utf-8')
    if f.startswith(u'\ufeff'):
        f = f.encode('utf8')[3:].decode('utf8')
    data = json.load(f)
    Temp_kb_dataLoc = []
    kb_id_group = []  # 放subject_id
    for idx, data_one in enumerate(data):
        lab_result = data_one['lab_result']
        for lab_result_one in lab_result:
            kb_id = lab_result_one['kb_id']
            if kb_id == -1:
                continue
            kb_data_one = {
                'kb_id': kb_id,
                'mention': lab_result_one['mention'].lower(),
                'stock_full_name': kb_dataNet[str(kb_id)]['stock_full_name'],
                'data': data_one['text'].lower()

            }
            if kb_id in kb_id_group:
                temp_kb_data_one = Temp_kb_dataLoc[kb_id_group.index(kb_id)]
                temp_kb_data_one['data'] += data_one['text'].lower()
            else:
                kb_id_group.append(kb_id)
                Temp_kb_dataLoc.append(kb_data_one)

    # 转换成kb_id索引
    kb_dataLoc = defaultdict(dict)
    for i in Temp_kb_dataLoc:
        kb_dataLoc.update({
            str(i['kb_id']): {
                'kb_id': str(i['kb_id']),
                'mention': i['mention'],
                'stock_full_name': i['stock_full_name'],
                'data': i['data']
            }
        })
    with open('./data_deal/kb_data/kb_dataLoc.pkl', 'wb') as f:
        pickle.dump(kb_dataLoc, f)

# print(kb_dataNet['317']['data'])
# print(kb_dataLoc['305'])
# 融合构成kb_data ，其中kb_dataNet中data为'NIL'值由取kb_dataLoc对应值填充（限制其长度低于512）
# 训练集中可能没有包含全部的id
if os.path.exists('./data_deal/kb_data/kb_data.pkl'):
    with open('./data_deal/kb_data/kb_data.pkl', 'rb') as f:
        kb_data = pickle.load(f)
else:
    kb_data = kb_dataNet
    for index in kb_data:
        kb_data_one = kb_data[index]
        if (kb_data_one['data'] == 'NIL' or len(kb_data_one['data']) <= 30):
            if bool(kb_dataLoc[index]):
                if kb_data_one['data'] == 'NIL':
                    temp_data = '该公司全名为' + kb_dataLoc[index]['stock_full_name'] + ',' + kb_dataLoc[index]['data']
                if len(kb_data_one['data']) <= 100:
                    temp_data = kb_data_one['data'] + kb_dataLoc[index]['data']
            else:  # NULL
                temp_data = '该公司全名为' + kb_data[index]['stock_full_name']

            kb_data_one['data'] = temp_data if len(temp_data) < Max_len else temp_data[:Max_len]
    with open('./data_deal/kb_data/kb_data.pkl', 'wb') as f:
        pickle.dump(kb_data, f)

# 构建实体词典，用于训练，根据"kb_id"检索
num_words = 3811
# 打开编码字典
with open('./data_deal/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

subject_data = kb_data  # c
for index in subject_data:
    i = kb_data[index]  # c
    data_seq = [word_index.get(c, num_words + 1) for c in i['data']]
    if len(data_seq) == 0:
        i['data'] = '无'
        data_seq = [0]
    i['data_seq'] = data_seq
# 保存字典
if os.path.exists('./data_deal/subject_data.pkl'):
    with open('./data_deal/subject_data.pkl', 'rb') as f:
        subject_data = pickle.load(f)
else:
    with open('./data_deal/subject_data.pkl', 'wb') as f:
        pickle.dump(subject_data, f)

# 构建实体词典，用于推断，根据"实体"检索
stockname_data = defaultdict(dict)
for index in kb_data:  # c
    i = kb_data[index]  # c
    data_seq = [word_index.get(c, num_words + 1) for c in i['data']]
    if len(data_seq) == 0:
        i['data'] = '无'
        data_seq = [0]
    stockname_data.update({
        i['mention']: {
            'kb_id': i['kb_id'],
            'stock_full_name': i['stock_full_name'],
            'data': i['data'],
            'data_seq': data_seq
        }
    })
# 保存字典
if os.path.exists('./data_deal/stockname_data.pkl'):
    with open('./data_deal/stockname_data.pkl', 'rb') as f:
        stockname_data = pickle.load(f)
else:
    with open('./data_deal/stockname_data.pkl', 'wb') as f:
        pickle.dump(stockname_data, f)
