# lic2019_ie
2019 Language and Intelligence Challenge, Information Extraction

[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/torch-1.0.0-brightgreen.svg)](https://pypi.org/project/torch/1.0.0)
[![](https://img.shields.io/badge/keras-2.2.4-brightgreen.svg)](https://pypi.org/project/keras/2.2.4)
[![](https://img.shields.io/badge/numpy-1.16.2-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.16.2)

## **项目简介**
2019语言与智能技术竞赛(2019 Language and Intelligence Challenge)，信息抽取赛道(Information Extraction),<http://lic2019.ccf.org.cn/kg>，竞赛代码分享。<br>

## **运行方式**
### **获取预训练数据**
1.词向量来源于<https://github.com/Embedding/Chinese-Word-Vectors>，在此表示感谢！<br><br>
2.运行脚本./embedding.py，抽取出字向量，保存在./data_deal中<br><br>

### **数据预处理**
1.运行脚本./preprocess.py在./data_deal中生成预处理结果，提供部分预处理数据，主要包含：<br>
* tokenizer.pkl，文本编码信息，推断test2数据会需要<br>
* tokenizer_pos.pkl，词性编码信息(数据中有缺失，用n替代)，推断test2数据会需要<br>
* p.pkl，spo关系编码，后续训练和推断都基于此<br>
* ./7000/xxx_data_process.pkl，预处理转换，后续训练和推断都基于此<br>

### **训练模型**
1.运行脚本./train.py，自行修改路径和epochs等参数<br><br>
2.模型和每次的预测结果都保存在./models/7000中，后续ensemble需要<br><br>

### **测试模型**
1.运行脚本./submit_dev.py，查看验证集ensemble情况<br><br>
2.运行脚本./submit_test.py，获取测试集ensemble结果用于提交(内含规则等，暂不公开)<br><br>

## **其他说明**
1.模型和苏剑林比赛开源的baseline一致，在参数完全相同、不使用预训练的情况下，苏剑林的模型(keras)f1=0.76，本模型(pytorch)f1=0.81，和部分赛友交流过差不多，原因不详，再次对苏剑林同学开源模型思路表示感谢；<br><br>
2.使用预训练词向量，百度百科效果最好，字保留数量=7000，我还使用了了人民日报和搜狗新闻，用于最后投票集成，单模得分详情见<https://github.com/renjunxiang/lic2019_ie/blob/master/other/evaluate.xlsx>；<br><br>
3.词性信息(POS)对实体有效截取有帮助，f1可以提高0.003-0.005，单模f1在0.83-0.84左右；<br><br>
4.集成投票对整体得分有显著提高(原则上是模型越多越能接近极限，由于时间原因最后只用了116个模型，1080ti*2推断需要60小时)，f1提高约0.02-0.03，极限水平约0.86；<br><br>
5.规则对整体得分有显著提高，f1提高约0.015，极限水平约0.875；<br><br>
6.尝试过ResNet、attention、transformer编码和BERT，效果比较差，大概是姿势不对QAQ；<br><br>
7.最后有一个不是很合适，但可以提高得分的方法，对未到投票阈值的spo，计算s、p、o等的频数、比例等信息，再训练一个二分类器(SVM，准确率约72%)，f1可以提高约0.0005-0.001<br><br>
PS:比赛代码修改后在服务器上执行，本地代码不保证没问题，如有问题请自行debug，再次对主办方开放数据集表示感谢！<br><br>
![](https://github.com/renjunxiang/lic2019_ie/blob/master/picture/score.png)