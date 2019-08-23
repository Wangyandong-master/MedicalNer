# MedicalNer
An easy-to-use named entity recognition (NER) for medical, implemented the LSTM+word-and-char-attention+CRF model in tensorflow.

## 1. Model
Bi-LSTM + word-and-char-attention + CRF，其中 word-and-char-attention通过参数门控机制将字词信息结合。

## 2. Usage
### 2.1 数据准备
训练数据处理成下列形式，特征之间用制表符(或空格)隔开，每行共n列，1至n-1列为特征，最后一列为label。

    入院 n O
    时间 n O
    ： x O
    2010 m O
    - x O
    05 m O
    14 m O
	15 m O
	: x O
	00 m O
	
    主 b O
	诉 vn O
	： x O
	发现 v O
	心脏 n B-身体部位
	杂音 n B-医学发现
	10 m B-时间词
	月余 n I-时间词
	。 x O
### 2.2 参数配置文件
模型参数设置均在hyperparams.py文件中，参数说明如下：

|  | 参数 |说明  |
| ------------ | ------------ | ------------ |
|1|word_embedding_size| 模型中词向量维度|
|2|char_embedding_size| 模型中词向量维度|
|3|kernels| CNN filter sizes, use window 3, 4, 5 for char CNN|
|4|char_hidden_size|cnn输出纬度 |
|5|lstm_hidden_size| lstm隐含层纬度|
|6|use_chars| 是否使用字特征|
|7|char_embedding_method| 字向量处理方式|
|8|use_crf| 是否使用crf|
|9|clip:|  梯度裁剪，用户设置，默认值`5`|
|10|use_char_attention| 是否使用字词attention进行信息结合|
|11|batch_size| ibatch size，用户设置`|
|12|num_epochs| 迭代次数，用户设置|
|13|dropout| dropout过拟合|
|14|learning_rate| 初始学习率`|
|15|learning_rate_decay| 学习率下降尺度|

### 2.3 预处理
    
预处理函数为main.py/build()函数, 主要功能，读取字词字典，读取字词向量。

### 2.４ 训练模型
运行main.py/run()函数

## 4. Requirements
-python3.6
-numpy
-tensorflow-gpu 1.1.0

说明：该项目是在windows平台操作，数据暂时未脱敏，只提供了一小部分，后期增加。

