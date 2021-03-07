# Text Mining Coursewrok 



## 开发须知

### 本项目命名规则

- 变量

  驼峰式命名，如：appleTree  

- 函数

  下划线式，如: apple_tree

- 文件名

  下划线式，如 apple_tree.py



### main方法

每个py文件中，计算过程需要函数封装，同时生成相应的main方法进行测试。



### mac和window用户需知

由于文件路径mac和linux用的都是正斜杆‘/’,而window是斜杆'\\', 所以，在运行时需要修改bowing.config和bilstm.config配置文件中的路径分隔符。同时question_classifier.py文件中开头对配置文件的路径读取也要进行相应修改。



### PYCHARM运行路径设置

由于TA在验证我们结果时使用的命令如下，因此确保项目的运行路径在TMCoursework1/src目录下(默认是TMCoursework1主目录下)。

```shell
% python3 question_classifier.py --train --config [configuration_file_path]
```



### 如何获取配置文件的参数

系统在question_classifier.py 的文件开头读取了config文件

```python
conf = ConfigParser()
conf.read('../data/bow.config')
```

如果需要获取某个设定值,在python文件中导入src.question_classifier文件里的conf,调用get 方法获取

```python
from src.question_classifier import conf
print(conf.get('param','model'))  # bow
```



## 文件

### 文件目前结构

> TMCoursework1

> > data

> > > dev.txt

> > > test.txt

> > > train.txt

> > > train_5500.txt

> > > bow.config							bow配置文件

> > > bilstm.config						bilstm配置文件

> > src

> > > question_classifier.py		程序主入口文件，捕获命令行参数

> > > pre_processing.py			 数据预处理文件

> > > split_file.py						 训练集验证集切分文件

> > > tokenization.py				  数据token化文件

> > > word_embeddings.py	   word embedding方法文件

> > > bag_of_words.py   			bag of words,输出句子vector

> > README.md

