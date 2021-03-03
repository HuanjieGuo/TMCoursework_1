# Text Mining Coursewrok  - 1

> 文件目前结构

- data
  - dev.txt
  - test.txt
  - train.txt
  - bow.config
  - bilstm.config
- src
  - question_classifier.py



> mac和window用户需知

由于文件路径mac和linux用的都是正斜杆‘/’,而window是斜杆'\\', 所以，在运行时需要修改bowing.config和bilstm.config配置文件中的路径分隔符。同时question_classifier.py文件中开头对配置文件的路径读取也要进行相应修改。

> 运行时需知

由于TA在验证我们结果时使用的命令如下，因此确保项目的运行路径在TMCoursework1/src目录下(默认是TMCoursework1主目录下)。

```shell
% python3 question_classifier.py --train --config [configuration_file_path]
```

> 如何获取配置文件的参数

由于在question_classifier.py 的文件开头读取了config文件，后面需要调用只需通过conf读取

```python
conf = ConfigParser()
conf.read('../data/bow.config')

# 读取配置文件中model的值
print(conf.get('param','model'))  # bow
```



## 函数介绍

> train_data切分函数

```python
def splitData():
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

def split(full_list, shuffle=False, ratio=0.2):
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

