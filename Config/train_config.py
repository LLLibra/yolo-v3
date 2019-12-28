# -*- coding:UTF-8 -*-
config = {
    "ModelName":"SSD300",
    "train_epoch":500,
    "train_batch_size":8,
    "lr":1e-4,                ##基础学习率
    "lr_decay_epoch":300,     ##多少次迭代 学习率衰减一次
    "save_epoch":300,         ##多少次迭代 保存一次模型
    "save_path":"",           ##模型保存路径
    "lr_decay_base_num":0.9,  ##学习率衰减基础参数
    "gpu_num":2,
    "gpus":[1,0,2,3],         ##使用哪些GPU
    "num_workers":2,
    "CUDA_VISIBLE":"0,1,2,3", ##看得到哪些GPU
    "weight_decay":1e-4,      ##损失函数正则化
    "Model_name":"",          ##想要保存的模型的名字
    "log_file_path":"Train_log/xx.txt",       ##日志文件的路径和名字
    "num_classes":81,        ##样本类别数
}