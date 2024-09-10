# 完整的项目结构 + 数据集的获取方式

```tree
├── dataset_verify.py
├── News Articles
│   ├── business
│   ├── entertainment
│   ├── politics
│   ├── sport
│   └── tech
├── Summaries
│   ├── business
│   ├── entertainment
│   ├── politics
│   ├── sport
│   └── tech
└── verify_failed_files.log
```

`News Articles` 和 `Summaries` 文件夹下分别有 `business`、`entertainment`、`politics`、`sport`、`tech` 五个子文件夹，每个子文件夹下存放着对应类别的文章和摘要。

数据的下载链接为：`https://www.kaggle.com/datasets/pariza/bbc-news-summary/data`

# 数据集读取

```python
from dataset_verify import final_dataset
train_set, test_set = final_dataset(shuffle=True, fold_index=0)
```

# 实验结果2024.9.10

代码: `log_reader.py`调用`read_log_print_markdown_table`函数。

| prefix | fold1 | fold2 | fold3 | fold4 | fold5 | average |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| bert-base-title-on-crf-on | 0.648 | 0.632 | 0.646 | 0.647 | 0.656 | 0.646 |
| roberta-base-title-off-crf-on | 0.811 | 0.809 | 0.799 | 0.803 | 0.808 | 0.806 |
| roberta-base-title-on-crf-off | 0.809 | 0.809 | 0.803 | 0.808 | 0.814 | 0.808 |
| roberta-base-title-on-crf-on | 0.816 | 0.809 | 0.802 | 0.805 | 0.812 | 0.809 |