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

> > > bow.config

> > > bilstm.config

> > src

> > > question_classifier.py

> > > ssentence_processing.py

> > > split_data.py
> >
> > README.md



### 文件介绍

#### question_classifier.py

本文件目前还没有实现函数，后面会用于设计TA调用的入口函数

```python
from configparser import ConfigParser

# conf can be used in other py file to get the parameters.
conf = ConfigParser()
conf.read('../data/bow.config')
```

#### sentence_processing.py

> sentenceProcessing函数

```python
def sentenceProcessing(file):
    labels = []
    sentences = []
    with open(file, 'r') as f:
        # result = f.read()
        # result = re.sub('[?]','',result)
        for line in f.readlines():
            line = line.strip('\n')
            # print(line)
            # print(line.split(' ',1))
            labels.append(line.split(' ', 1)[0])
            sentences.append(line.split(' ', 1)[1])
    return labels,sentences
```

本方法用于**分割**label和sentence, 最终把分割后的两个数组返回。

> lower_first_letter函数

```python
# 首字母小写
def lower_first_letter(sentences):
    for i in range(len(sentences)):
        tmp = list(sentences[i])
        tmp[0] = tmp[0].lower()
        sentences[i] = ''.join(tmp)
    return sentences
```

用于将每个句子中的首字母小写后返回。

#### split_file.py

> get_train_dev函数

```python
def get_train_dev():
    path = os.path.join(os.getcwd(),"..","data", "train_5500.txt")
    f = open(path)
    lines = f.readlines()
    f.close()
    train, dev = split(lines, shuffle=True, ratio=0.9)

    # read
    file = open(conf.get('param','path_train'), 'w')
    for i in range(len(train)):
        file.write(train[i])
    file.close()

    file = open(conf.get('param','path_dev'), 'w')
    for i in range(len(dev)):
        file.write(dev[i])
    file.close()
```

用于将train_5500.txt随机切分为9:1结构，并生成train.txt和dev.txt

> random_split函数

```python
def random_split(full_list, shuffle=False, ratio=0):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2
```

随机切分函数，供get_train_dev函数调用。