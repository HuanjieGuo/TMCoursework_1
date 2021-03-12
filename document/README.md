<<<<<<< HEAD
### How to run

```shell
cd src

# make sure that you are in the src folder

# train the bow model, you would wait for several minutes here 
python3 question_classifier.py --train --config '../data/bow.config'

# test the bow model
python3 question_classifier.py --test --config '../data/bow.config'

# train the bilstm model
python3 question_classifier.py --train --config '../data/bilstm.config'

# test the bilstm model
python3 question_classifier.py --test --config '../data/bilstm.config'
```



### How to change the params?

> bilstm.config

```shell
# if pre_train is False, randomly initialized
pre_train: True

# freeze or fine- tune 
freeze : False
```



> bow.config

```shell
# 'randomly' or 'pre_train'
word_embedding_type = randomly

# learning rate
lr_param : 0.02

# freeze or fine- tune 
freeze : False
```

=======
### How to run

```shell
cd src

# make sure that you are in the src folder

# train the bow model, you would wait for several minutes here 
python3 question_classifier.py --train --config '../data/bow.config'

# test the bow model
python3 question_classifier.py --test --config '../data/bow.config'

# train the bilstm model
python3 question_classifier.py --train --config '../data/bilstm.config'

# test the bilstm model
python3 question_classifier.py --test --config '../data/bilstm.config'
```



### How to change the params?

> bilstm.config

```shell
# if pre_train is False, randomly initialized
pre_train: True

# freeze or fine- tune 
freeze : False
```



> bow.config

```shell
# 'randomly' or 'pre_train'
word_embedding_type = randomly

# learning rate
lr_param : 0.02

# freeze or fine- tune 
freeze : False
```

>>>>>>> 8fcf83eddea79ded1dc918b49235096d56ae26c7
