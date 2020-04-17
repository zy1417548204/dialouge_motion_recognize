#!/bin/bash

#-----说明--------
#sed 追加文件内容
#1、a 在匹配行后面追加
#2、i 在匹配行前面追加
#3、r 将文件内容追加到匹配行后面
#4、w 将匹配行写入指定文件

#-----说明--------

# 修改配置，使用预训练好的模型
!cd /home/aistudio/work/ && sed -i '7c MODEL_PATH=./textcnn' run.sh
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"textcnn_net",#' config.json

# 模型预测，并查看结果
!cd /home/aistudio/work/ && sh run.sh infer


# 修改配置，选择cnn模型
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"cnn_net",#' config.json
!cd /home/aistudio/work/ && sed -i 's#"init_checkpoint":.*$#"init_checkpoint":"",#' config.json

# 修改训练后模型保存的路径
!cd /home/aistudio/work/ && sed -i '6c CKPT_PATH=./save_models/cnn' run.sh

# 模型训练
!cd /home/aistudio/work/ && sh run.sh train


# 确保使用的模型为CNN
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"cnn_net",#' config.json
# 使用刚才训练的cnn模型
!cd /home/aistudio/work/ && sed -i '7c MODEL_PATH=./save_models/cnn/step_756' run.sh

# 模型评估
!cd /home/aistudio/work/ && sh run.sh eval

# 查看预测的数据
!cat /home/aistudio/data/data12605/data/infer.txt

# 使用刚才训练的cnn模型
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"cnn_net",#' config.json
!cd /home/aistudio/work/ && sed -i '7c MODEL_PATH=./save_models/cnn/step_756' run.sh
# 模型预测
!cd /home/aistudio/work/ && sh run.sh infer


# 更改模型为TextCNN
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"textcnn_net",#' config.json
!cd /home/aistudio/work/ && sed -i 's#"init_checkpoint":.*$#"init_checkpoint":"",#' config.json
# 修改模型保存目录
!cd /home/aistudio/work/ && sed -i '6c CKPT_PATH=./save_models/textcnn' run.sh
# 模型训练
!cd /home/aistudio/work/ && sh run.sh train
# 使用上面训练好的textcnn模型
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"textcnn_net",#' config.json
!cd /home/aistudio/work/ && sed -i '7c MODEL_PATH=./save_models/textcnn/step_756' run.sh
# 模型评估
!cd /home/aistudio/work/ && sh run.sh eval


#基于预训练的TextCNN进行Finetune
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"textcnn_net",#' config.json
# 使用预训练的textcnn模型
!cd /home/aistudio/work/ && sed -i 's#"init_checkpoint":.*$#"init_checkpoint":"./textcnn",#' config.json
# 修改学习率和保存的模型目录
!cd /home/aistudio/work/ && sed -i 's#"lr":.*$#"lr":0.0001,#' config.json
!cd /home/aistudio/work/ && sed -i '6c CKPT_PATH=./save_models/textcnn_finetune' run.sh
# 模型训练
!cd /home/aistudio/work/ && sh run.sh train
# 修改配置，使用上面训练得到的模型
!cd /home/aistudio/work/ && sed -i 's#"model_type":.*$#"model_type":"textcnn_net",#' config.json
!cd /home/aistudio/work/ && sed -i '7c MODEL_PATH=./save_models/textcnn_finetune/step_756' run.sh
# 模型评估
!cd /home/aistudio/work/ && sh run.sh eval


#基于ERNIE模型进行Finetune
!cd /home/aistudio/work/ && mkdir -p pretrain_models/ernie
%cd /home/aistudio/work/pretrain_models/ernie
# 获取ernie预训练模型
!wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz -O ERNIE_stable-1.0.1.tar.gz
!tar -zxvf ERNIE_stable-1.0.1.tar.gz && rm ERNIE_stable-1.0.1.tar.gz
# 基于ERNIE模型finetune训练
!cd /home/aistudio/work/ && sh run_ernie.sh train
# 模型评估
!cd /home/aistudio/work/ && sh run_ernie.sh eval
# 查看所有参数及说明
!cd /home/aistudio/work/ && python run_classifier.py -h

# 解压分词工具包，对测试数据进行分词
!cd /home/aistudio/work/ && unzip -qo tokenizer.zip
!cd /home/aistudio/work/tokenizer && python tokenizer.py --test_data_dir test.txt.utf8 > new_query.txt
# 查看分词结果
!cd /home/aistudio/work/tokenizer && cat new_query.txt



