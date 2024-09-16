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

# t_test结果

`/home/taku/Documents/auto_japanese_emphasis/archive/BBC News Summary/t_test_results/t_test_scores_20240910_141002.json`

|         | roberta-base-title-on-crf-on (0.876) | roberta-base-title-on-crf-off (0.887) | roberta-base-title-off-crf-on (0.884) | bert-base-title-on-crf-on (0.732) |
|---------|---------|---------|---------|---------|
| roberta-base-title-on-crf-on (0.876) | - | 0.000 | 0.008 | 0.000 |
| roberta-base-title-on-crf-off (0.887) | 0.000 | - | 0.510 | 0.000 |
| roberta-base-title-off-crf-on (0.884) | 0.008 | 0.510 | - | 0.000 |
| bert-base-title-on-crf-on (0.732) | 0.000 | 0.000 | 0.000 | - |

# 2024.9.11 t_test_add_modules结果

`t_test.py` > `dic = load_score_dict('t_test_scores_20240911_102321.json')` > `print_p_value(dic)`

|         | bert_vanilla (0.648) | bert_crf_on (0.740) | bert_title_on (0.741) | bert-base-title-on-crf-on (0.732) | roberta_vanilla (0.885) | roberta-base-title-on-crf-on (0.890) |
|---------|---------|---------|---------|---------|---------|---------|
| bert_vanilla (0.648) | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bert_crf_on (0.740) | 0.000 | - | 0.956 | 0.051 | 0.000 | 0.000 |
| bert_title_on (0.741) | 0.000 | 0.956 | - | 0.069 | 0.000 | 0.000 |
| bert-base-title-on-crf-on (0.732) | 0.000 | 0.051 | 0.069 | - | 0.000 | 0.000 |
| roberta_vanilla (0.885) | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.051 |
| roberta-base-title-on-crf-on (0.890) | 0.000 | 0.000 | 0.000 | 0.000 | 0.051 | - |

# 2024.9.11 正常输出的结果

| prefix | fold1 | fold2 | fold3 | fold4 | fold5 | average |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| bert_vanilla | 0.654 | 0.642 | 0.651 | 0.652 | 0.643 | 0.648 |
| bert_crf_on | 0.659 | 0.647 | 0.683 | 0.654 | 0.650 | 0.659 |
| bert_title_on | 0.667 | 0.644 | 0.672 | 0.640 | 0.652 | 0.655 |
| roberta_vanilla | 0.808 | 0.805 | 0.808 | 0.811 | 0.804 | 0.807 |
| roberta-base-title-on-crf-on | 0.816 | 0.809 | 0.802 | 0.805 | 0.812 | 0.809 |

# 2024.9.11 正常输出的结果

| prefix | fold1 | fold2 | fold3 | fold4 | fold5 | average |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| bert_vanilla | 0.712 | 0.672 | 0.704 | 0.760 | 0.694 | 0.708 |
| bert_crf_on | 0.706 | 0.684 | 0.691 | 0.714 | 0.715 | 0.702 |
| bert_title_on | 0.673 | 0.670 | 0.687 | 0.688 | 0.689 | 0.681 |
| roberta_vanilla | 0.818 | 0.806 | 0.802 | 0.816 | 0.804 | 0.809 |
| roberta-base-title-on-crf-on | 0.813 | 0.807 | 0.810 | 0.818 | 0.811 | 0.812 |


[[0.70853673 0.6206482  0.64943526]
 [0.88011859 0.84863036 0.85932743]
 [0.81991531 0.76991621 0.78596313]
 [0.92654656 0.91065957 0.91617246]
 [0.93363342 0.91853627 0.92414469]]

 | prefix | fold1 | fold2 | fold3 | fold4 | fold5 | average |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| bert_vanilla | 0.712 | 0.672 | 0.704 | 0.760 | 0.694 | 0.708 |
| bert_crf_on | 0.706 | 0.684 | 0.691 | 0.714 | 0.715 | 0.702 |
| bert_title_on | 0.673 | 0.670 | 0.687 | 0.688 | 0.689 | 0.681 |
| roberta_vanilla | 0.818 | 0.806 | 0.802 | 0.816 | 0.804 | 0.809 |
| roberta-base-title-on-crf-on | 0.813 | 0.807 | 0.810 | 0.818 | 0.811 | 0.812 |


# 2024.9.11 t_test_add_modules结果

|         | bert_vanilla (0.709) | bert_crf_on (0.880) | bert_title_on (0.820) | roberta_vanilla (0.927) | roberta-base-title-on-crf-on (0.934) |
|---------|---------|---------|---------|---------|---------|
| bert_vanilla (0.709) | - | 0.000 | 0.000 | 0.000 | 0.000 |
| bert_crf_on (0.880) | 0.000 | - | 0.000 | 0.000 | 0.000 |
| bert_title_on (0.820) | 0.000 | 0.000 | - | 0.000 | 0.000 |
| roberta_vanilla (0.927) | 0.000 | 0.000 | 0.000 | - | 0.000 |
| roberta-base-title-on-crf-on (0.934) | 0.000 | 0.000 | 0.000 | 0.000 | - |


|         | bert_vanilla (0.709) | bert_crf_on (0.880) | bert_title_on (0.820) | roberta_vanilla (0.927) | roberta-base-title-on-crf-on (0.934) |
|---------|---------|---------|---------|---------|---------|
| bert_vanilla (0.709) | - | 2.1110830837248416e-243 | 3.8046269237547824e-137 | 0.0 | 0.0 |
| bert_crf_on (0.880) | 2.1110830837248416e-243 | - | 1.58150376287317e-53 | 2.3735910432736928e-37 | 4.978226587681683e-53 |
| bert_title_on (0.820) | 3.8046269237547824e-137 | 1.58150376287317e-53 | - | 5.131848678897256e-144 | 1.3307348096356468e-158 |
| roberta_vanilla (0.927) | 0.0 | 2.3735910432736928e-37 | 5.131848678897256e-144 | - | 0.00030840127984713763 |
| roberta-base-title-on-crf-on (0.934) | 0.0 | 4.978226587681683e-53 | 1.3307348096356468e-158 | 0.00030840127984713763 | - |