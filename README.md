# 目录

- [目录](#目录)
- [1. 模型介绍](#1-模型介绍)
  - [1.1. 网络模型结构](#11-网络模型结构)
  - [1.2. 数据集](#12-数据集)
- [2. 代码目录结构说明](#2-代码目录结构说明)
  - [2.1. 脚本参数](#21-脚本参数)
- [3. 自验结果](#3-自验结果)
  - [3.1. 自验环境](#31-自验环境)
  - [3.2. 训练超参数](#32-训练超参数)
  - [3.3. 训练](#33-训练)
    - [3.3.1. 训练之前](#331-训练之前)
    - [3.3.2. 启动训练脚本](#332-启动训练脚本)
  - [3.4. 评估过程](#34-评估过程)
    - [3.4.1. 启动评估脚本](#341-启动评估脚本)
    - [3.4.2. 评估精度结果](#342-评估精度结果)
- [4. 参考资料](#4-参考资料)
  - [4.1. 参考论文](#41-参考论文)
  - [4.2. 参考git项目](#42-参考git项目)

# [1. 模型介绍](#contents)

TransE、TransH、TransR、TransD是知识图谱嵌入的模型。此模型的"知识"表示为一个三元组(头部,关系,尾部)，其中头部和尾部是实体。

**TransE** 模型的基本思想是使头部向量和关系向量之和与尾部向量尽可能接近。
距离采用L1或L2范数计算。用于训练该模型的损失函数是根据正负样本得分计算得到的差值损失。负采样通过替换原始三元组中的头部或尾部实体来执行。该模型有利于管理一对一的关系。

**TransH** 允许我们处理一对多、多对一和多对多关系的问题。它的基本思想是将关系重新解释为超平面上的平移。

**TransR** 的基本思想是实体和关系可以有不同的语义空间。
它使用可训练的投影矩阵将实体投影到多关系空间。
它也存在一些不足。例如，投影矩阵仅由关系决定，并且假定头部和尾部来自同一语义空间。此外，**TransR** 模型的参数量较大，不适用于大规模任务。

**TransD** 通过使用头部和尾部实体的动态映射来弥补 **TransD** 模型的缺陷。
头部和尾部的投影矩阵分别由头部关系和尾部关系对计算得出。

## [1.1. 网络模型结构](#contents)

所有模型的基本元素都是产生嵌入的实体和关系的可训练查找表。

## [1.2. 数据集](#contents)

我们使用 **Wordnet** 和 **Fresbase** 数据集对模型进行训练。

预处理的数据文件可通过以下链接获取：

- [WN18RR (Wordnet)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/WN18RR)
- [FB15K237 (Freebase)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/FB15K237)

# [2. 代码目录结构说明](#contents)

```text
./Fast-TransX
├── configs  # 模型配置文件
│   ├── default_config.yaml
│   ├── transD_FB15K_config.yaml
│   ├── transD_WN18_config.yaml
│   ├── transE_FB15K_config.yaml
│   ├── transE_WN18_config.yaml
│   ├── transH_FB15K_config.yaml
│   ├── transH_WN18_config.yaml
│   ├── transR_FB15K_config.yaml
│   └── transR_WN18_config.yaml
├── dataset   # 数据集
│ ├── FB15K
│ └── WN18
├── model_utils  # Model Arts通用工具
│   ├── config.py
│   ├── device_adapter.py
│   ├── __init__.py
│   ├── local_adapter.py
│   └── moxing_adapter.py
├── scripts  # 进行训练和评估的Shell脚本
│   ├── run_eval_npu.sh
│   ├── run_export_npu.sh
│   └── run_train_npu.sh
├── src
│   ├── base  # 数据集的 C++ 后端代码
│   │   ├── Base.cpp
│   │   ├── CMakeLists.txt
│   │   ├── Corrupt.h
│   │   ├── Random.h
│   │   ├── Reader.h
│   │   ├── Setting.h
│   │   └── Triple.h
│   ├── dataset_lib  # 已编译的数据集工具
│   │   └── train_dataset_lib.so
│   ├── utils
│   │   └── logging.py  # 日志记录程序
│   ├── dataset.py
│   ├── loss.py
│   ├── make.sh
│   ├── metric.py
│   ├── model_builder.py  # 脚本构建模型
│   ├── trans_x.py  # 模型定义
│   └── __init__.py
├── eval.py  # 对完成训练的模型进行评估的脚本
├── export.py  # 导出完成训练的模型的脚本
├── requirements.txt  # 附加依赖
├── train.py  # 开始训练的脚本
└── README.md  # 说明文档
```

## [2.1. 脚本参数](#contents)

训练和评估的参数可以通过 **\* . yaml**配置文件提供，也可以直接向 train.py, eval.py and export.py 脚本提供参数。

```yaml
device_target: "Ascend"         # 仅使用Ascend NPU进行测试
is_train_distributed: False  # 是否使用NCCL进行多NPU训练
group_size: 1                # 设备数量
device_id: 0                 # 设备ID (仅针对单个设备训练)
seed: 1                      # Random seed

# Model options
model_name: "TransE"         # 模型名称 (TransE / TransH / TransR / TransD)
dim_e: 50                    # 实体嵌入大小
dim_r: 50                    # 关系嵌入大小

# 数据集选项
dataset_root: "/path/to/dataset/root"
train_triplet_file_name: "train2id.txt"
eval_triplet_file_name: "test2id.txt"
filter_triplets_files_names:  # 具有正三元组样本的文件
  - "train2id.txt"
  - "valid2id.txt"
  - "test2id.txt"
entities_file_name: "entity2id.txt"
relations_file_name: "relation2id.txt"
negative_sampling_rate: 1    # 单个正样本中负样本的数量。
train_batch_size: 868

# 日志选项
train_output_dir: "train-outputs/"
eval_output_dir: "eval-output/"
export_output_dir: "export-output/"
ckpt_save_interval: 5
ckpt_save_on_master_only: True
keep_checkpoint_max: 10
log_interval: 100

# 训练选项
pre_trained: ""              # 预训练模型的路径（TransD需要）
lr: 0.5                      # 学习率
epochs_num: 1000             # epochs的数值
weight_decay: 0.0            # 权衰减
margin: 6.0                  # 间隔损失的参数
train_use_data_sink: False

# 评估和导出选项
ckpt_file: "/path/to/trained/checkpoint"
file_format: "MINDIR"
eval_use_data_sink: False
export_batch_size: 1000      # 导出模型的batch size
```

# [3. 自验结果](contents)

## [3.1. 自验环境](contents)

- 硬件环境
  - CPU：aarch64  192核 
  - NPU：910ProA
- MindSpore version:  1.5.1
- Python
  - 版本：Python 3.7.6
  - 第三方库和依赖：requirements.txt

## [3.2. 训练超参数](contents)

| Parameters    |         |         |         |         |          |          |          |          |
| ------------- | ------- | ------- | ------- | ------- | -------- | -------- | -------- | -------- |
| Model         | TransE  | TransH  | TransR  | TransD  | TransE   | TransH   | TransR   | TransD   |
| Dataset       | Wordnet | Wordnet | Wordnet | Wordnet | Freebase | Freebase | Freebase | Freebase |
| Batch size    | 868     | 868     | 868     | 868     | 2721     | 2721     | 453      | 2721     |
| Learning rate | 0.5     | 0.5     | 0.05    | 0.5     | 1        | 0.5      | 0.16667  | 1        |
| Epochs        | 1000    | 300     | 250     | 200     | 1000     | 1000     | 1000     | 1000     |

Loss function : TripletsMarginLoss()

Optimizer : opt

## [3.3. 训练](contents)

### [3.3.1. 训练之前](#contents)

需要编译库来生成 corrupted triplets

SOTA实现使用三元组过滤，以确保 corrupted triplets 不会出现在 original triplets 中。

这个过滤过程很难矢量化的在Python中有效实现，所以我们使用我们自定义的 **\*.so** 库。

请到 **./transX/src** 目录下构建库并且执行：

```shell script
bash make.sh
```

构建成功后,   在**./transX/src/dataset_lib**.会出现**train_dataset_lib.so**

### [3.3.2. 启动训练脚本](contents)

你可以通过运行python脚本启动NPU进行训练：

- 没有预训练的模型

  ```shell script
  python train.py --config_path=/parth/to/model_config.yaml --dataset_root=/path/to/dataset
  ```

- 有预训练的模型

  ```shell script
  python train.py --config_path=/parth/to/model_config.yaml --dataset_root=/path/to/dataset --pre_trained=/path/to/pretrain.ckpt
  ```

或者通过Shell脚本进行训练（**推荐**）：

- 没有预训练模型

  ```shell script
  bash scripts/run_train_npu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME]
  ```

- 有预训练模型

  ```shell script
  bash scripts/run_train_npu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME] [PRETRAIN_CKPT]
  ```

> DATASET_NAME 必须为 "WN18" or "FB15K"
>
> MODEL_NAME 必须为 "TransE", "TransH", "TransR" 或 "TransD"
>
> 使用该名称将选择 ./configs 目录中相应的配置文件。

训练结果将会存放在 **./train-outputs** 目录下
如果使用 Shell 脚本, 日志信息将会重定向到 **./train-logs** 目录下

## [3.4. 评估过程](#contents)

### [3.4.1. 启动评估脚本](contents)

你可以通过运行以下 python 脚本来开始评估：

```shell script
python eval.py --config_path=/parth/to/model_config.yaml --dataset_root=/path/to/dataset --ckpt_file=/path/to/trained.ckpt
```

或者 Shell 脚本（**推荐**）

```shell script
bash scripts/run_eval_npu.sh [DATASET_ROOT] [DATASET_NAME] [MODEL_NAME] [CKPT_PATH]
```

> DATASET_NAME 必须为 "WN18" or "FB15K"
>
> MODEL_NAME 必须为 "TransE", "TransH", "TransR" 或 "TransD"
>
> 使用该名称将选择 ./configs 目录中相应的配置文件。

#### [3.4.2. 评估精度结果](contents)

|                   |         |         |         |         |          |          |          |          |
| ----------------- | ------- | ------- | ------- | ------- | -------- | -------- | -------- | -------- |
| Model             | TransE  | TransH  | TransR  | TransD  | TransE   | TransH   | TransR   | TransD   |
| Dataset           | Wordnet | Wordnet | Wordnet | Wordnet | Freebase | Freebase | Freebase | Freebase |
| Accuracy (Hit@10) | 0.8674  | 0.8591  | 0.8941  | 0.9064  | 0.7351   | 0.7733   | 0.8013   | 0.7836   |



# [4. 参考资料](#contents)

## [4.1. 参考论文](contents)

- [Paper TransE](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
  Translating Embeddings for Modeling Multi-relational Data（2013)
- [Paper TransH](https://persagen.com/files/misc/wang2014knowledge.pdf)
  Knowledge Graph Embedding by Translating on Hyperplanes（2014)
- [Paper TransR (download)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwicz7i6hvfzAhVEmYsKHR8qClYQFnoECAQQAQ&url=https%3A%2F%2Fwww.aaai.org%2Focs%2Findex.php%2FAAAI%2FAAAI15%2Fpaper%2Fdownload%2F9571%2F9523&usg=AOvVaw07cpMPMew9IF8Yn5iZDvCu)
  Learning Entity and Relation Embeddings for Knowledge Graph Completion（2015
- [Paper TransD](https://aclanthology.org/P15-1067.pdf)
  Knowledge Graph Embedding via Dynamic Mapping Matrix（2015

## [4.2. 参考git项目](contents)

- https://gitee.com/mindspore/mindspore.git
- https://github.com/thunlp/Fast-TransX.git

- https://github.com/yangyucheng000/transX.git
- https://github.com/thunlp/OpenKE.git
