import pickle
import pickle
import sys

path = ''
path = sys.argv[1]
sys.path.append('%s' % path)
print(path)
import os
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from net import MyDataset, collate_fn, collate_fn_link, deal_eval, seqs2batch, dataset
from net import Net
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel
import argparse

cuda_num = '0'  # cuda:0/1
Bert_name = 'bert'  # roberta_wwm,wwm,ernie
loss_weight_ = 2  # 3/2
num_layers = 3  # 3/4
hidden_dim = 768  # 768/1024
BS = 64  # BitchSize
num_words_ = 3811
max_len_ = 400  # 300/400/500
EMBEDDING_DIM = 300
epochs = 15
ner_id = 13  # Best ner_model
k = 0.8500
lr = 0.001  # Learning rate

dataset.device = "cuda:%s" % cuda_num
device = dataset.device
# torch.manual_seed(1)


with open('./data_deal/weight_baidubaike.pkl', 'rb') as f:
    embedding = pickle.load(f)
    embedding = torch.FloatTensor(embedding).to(device)

# 文本编码、词典
with open('./data_deal/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

# 训练集
with open('./data_deal/train_net_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

# 根据"kb_id"检索的实体词典
with open('./data_deal/subject_data.pkl', 'rb') as f:
    subject_data = pickle.load(f)

# 根据"实体"检索的实体词典
with open('./data_deal/stockname_data.pkl', 'rb') as f:
    stockname_data = pickle.load(f)

# 读取验证集预处理
with open('./data_deal/dev_data.pkl', 'rb') as f:
    develop_data = pickle.load(f)

# 拆分训练集
train1_data, train2_data = train_test_split(train_data,
                                            test_size=0.1,
                                            random_state=1)
trainloader1 = torch.utils.data.DataLoader(
    dataset=MyDataset(train1_data, subject_data, stockname_data),
    batch_size=BS, shuffle=True, collate_fn=collate_fn_link)

for num_words in [num_words_]:
    for max_len in [max_len_]:
        for embedding_name in [Bert_name]:  # ['roberta_wwm','wwm','ernie']
            bert_path = './Bert/' + embedding_name + '/'
            dataset.tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')
            dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
            dataset.BERT.eval()
            dataset.max_len = max_len
            for loss_weight in [loss_weight_]:
                accu_ = 0
                while accu_ < k:
                    # vocab_size有pad和unknow，维度+2
                    model = Net(vocab_size=len(word_index) + 2,
                                embedding_dim=EMBEDDING_DIM,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim,
                                embedding=embedding,
                                device=device).to(device)

                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    # ==========导入ner预训练结果==========
                    ner_model = 'lstm_%d_%d_%d' % (num_layers, hidden_dim, loss_weight)
                    checkpoint = torch.load('./results_ner/%s/%s/new_%03d.pth' % (
                        embedding_name, ner_model, ner_id), map_location=device)

                    # 导入ner部分
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)

                    # ===================================

                    file_name = 'lstm_%d_%d_%d_len_%d_lf_2_l_2' % (
                        num_layers, hidden_dim, loss_weight, max_len)

                    if not os.path.exists('./results/%s/%s/' % (embedding_name, file_name)):
                        os.makedirs('./results/%s/%s/' % (embedding_name, file_name))

                    score1 = []
                    result1 = []
                    for epoch in range(epochs):
                        print('Start Epoch: %d\n' % (epoch + 1))
                        sum_link_loss = 0.0
                        model.train()
                        for i, data in enumerate(trainloader1):
                            data_ner, data_link = data
                            # 训练link
                            text_seqs_link, kb_seqs_link, labels_link = data_link
                            nums = (len(text_seqs_link) - 1) // BS + 1
                            for n in range(nums):
                                optimizer.zero_grad()
                                text_seqs, _ = seqs2batch(text_seqs_link[(n * BS):(n * BS + BS)])
                                text_seqs = torch.LongTensor(text_seqs).to(device)
                                kb_seqs, _ = seqs2batch(kb_seqs_link[(n * BS):(n * BS + BS)])
                                kb_seqs = torch.LongTensor(kb_seqs).to(device)
                                link_labels = torch.Tensor(labels_link[(n * BS):(n * BS + BS)]).to(device)

                                # link损失
                                link_loss = model.cal_link_loss(text_seqs,
                                                                kb_seqs,
                                                                link_labels)
                                link_loss.backward()
                                optimizer.step()
                                sum_link_loss += link_loss.item() / nums

                            if (i + 1) % 200 == 0:
                                print('\nEpoch: %d ,batch: %d' % (epoch + 1, i + 1))
                                print('link_loss: %f' % (sum_link_loss / 200))
                                sum_link_loss = 0.0

                        # train2得分=====================================================================
                        model.eval()
                        p_len = 0.001
                        l_len = 0.001
                        correct_len = 0.001

                        p_len1 = 0.001
                        l_len1 = 0.001
                        correct_len1 = 0.001

                        for idx, data in enumerate(train2_data):

                            # kb_id = list(j[0] for j in data['entity_list'])[0]
                            # if kb_id == -1:  # 将非领域词剔除掉
                            #     continue

                            model.zero_grad()
                            text_seqs = deal_eval([data])
                            text_seqs = text_seqs.to(device)
                            text = data['text']
                            with torch.no_grad():
                                entity_predict = model(text_seqs,
                                                       text,
                                                       stockname_data,
                                                       None)

                            p_set = set([j[:-1] for j in entity_predict])
                            p_len += len(p_set)
                            l_set = set(data['entity_list'])
                            l_len += len(l_set)
                            correct_len += len(p_set.intersection(l_set))

                            p_set1 = set([j[1:-1] for j in entity_predict])
                            p_len1 += len(p_set1)
                            l_set1 = set([j[1:] for j in data['entity_list']])
                            l_len1 += len(l_set1)
                            correct_len1 += len(p_set1.intersection(l_set1))

                            if (idx + 1) % 2000 == 0:
                                print('finish train_2 %d' % (idx + 1))

                        Precision = correct_len / p_len
                        Recall = correct_len / l_len
                        F1 = 2 * Precision * Recall / (Precision + Recall)

                        Precision1 = correct_len1 / p_len1
                        Recall1 = correct_len1 / l_len1
                        F1_1 = 2 * Precision1 * Recall1 / (Precision1 + Recall1)

                        accu = F1 / F1_1
                        score1.append([epoch + 1,
                                       round(Precision1, 4), round(Recall1, 4), round(F1_1, 4), round(accu, 4),
                                       round(Precision, 4), round(Recall, 4), round(F1, 4)])
                        print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

                        score1_df = pd.DataFrame(score1,
                                                 columns=['Epoch',
                                                          'P_n', 'R_n', 'F_n', 'link_accu',
                                                          'P', 'R', 'F1'])
                        print(score1_df)
                        score1_df.to_csv('./results/%s/%s/new_train_2.csv' % (embedding_name, file_name),
                                         index=False)

                        accu_ = max(accu_, accu)
                        if accu >= k:
                            # 保存网络参数
                            torch.save(model.state_dict(),
                                       './results/%s/%s/new_param_%03d.pth' % (
                                           embedding_name, file_name, epoch + 1))

                            torch.save(model,
                                       './results/%s/%s/new_%03d.pth' % (
                                           embedding_name, file_name, epoch + 1))

                            # eval预测结果=====================================================================
                            #
                            model.eval()
                            p_len = 0.001
                            l_len = 0.001
                            correct_len = 0.001
                            p_len1 = 0.001
                            l_len1 = 0.001
                            correct_len1 = 0.001
                            for idx, data in enumerate(develop_data):

                                # kb_id = list(j[0] for j in data['entity_list'])[0]
                                # if kb_id == -1:  # 将非领域词剔除掉
                                #     continue

                                model.zero_grad()
                                text_seq = deal_eval([data])
                                text_seq = text_seq.to(device)
                                text = data['text']
                                with torch.no_grad():
                                    entity_predict = model(text_seq, text, stockname_data, None)

                                # 同
                                p_set = set([j[:-1] for j in entity_predict])
                                p_len += len(p_set)
                                l_set = set(data['entity_list'])
                                l_len += len(l_set)
                                correct_len += len(p_set.intersection(l_set))

                                p_set1 = set([j[1:-1] for j in entity_predict])
                                p_len1 += len(p_set1)
                                l_set1 = set([j[1:] for j in data['entity_list']])
                                l_len1 += len(l_set1)
                                correct_len1 += len(p_set1.intersection(l_set1))
                                #
                                if (idx + 1) % 1000 == 0:
                                    print('finish dev %d' % (idx + 1))
                            # 同
                            Precision = correct_len / p_len
                            Recall = correct_len / l_len
                            F1 = 2 * Precision * Recall / (Precision + Recall)

                            Precision1 = correct_len1 / p_len1
                            Recall1 = correct_len1 / l_len1
                            F1_1 = 2 * Precision1 * Recall1 / (Precision1 + Recall1)

                            accu = F1 / F1_1
                            result1.append([epoch + 1,
                                            round(Precision1, 4), round(Recall1, 4), round(F1_1, 4), round(accu, 4),
                                            round(Precision, 4), round(Recall, 4), round(F1, 4)])
                            print('验证集结果：')
                            print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

                            result1_df = pd.DataFrame(result1,
                                                      columns=['Epoch',
                                                               'P_n', 'R_n', 'F_n', 'link_accu',
                                                               'P', 'R', 'F1'])

                            print(result1_df)
                            result1_df.to_csv('./results/%s/%s/new_train_2_result.csv' % (embedding_name, file_name),
                                              index=False)

                            accu_ = max(accu_, accu)
