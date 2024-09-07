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