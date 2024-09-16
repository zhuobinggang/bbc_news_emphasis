def main():
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot, Rouge_Logger
    from main import Sector_2024

    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)

    # 初始化模型
    model_wrapper = ModelWrapper(Sector_2024())
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())  # 添加日志记录器
    # 训练模型并打印结果
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=4, check_step=40, total_step=1200, logger=logger)  # 添加logger参数

def test_rouge1():
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot
    from main import Sector_2024
    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)
    model_wrapper = ModelWrapper(Sector_2024())
    model_wrapper.set_meta(fold_index=0, repeat_index=0)
    model_wrapper.load_checkpoint()
    scores = model_wrapper.calc_rouge1(test_set)
    # 输出平均precision, recall, f-score
    print(f'precision: {sum([score.precision for score in scores]) / len(scores):.2f}')
    print(f'recall: {sum([score.recall for score in scores]) / len(scores):.2f}')
    print(f'f-score: {sum([score.fmeasure for score in scores]) / len(scores):.2f}')


def test_rouges():
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot
    from main import Sector_2024
    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)
    model_wrapper = ModelWrapper(Sector_2024())
    model_wrapper.set_meta(fold_index=0, repeat_index=0)
    model_wrapper.load_checkpoint()
    scores = model_wrapper.calc_rouges(test_set)
    # 输出rouge1, rouge2, rougeL的平均precision, recall, f-score
    print(f'rouge1 precision: {sum([score["rouge1"].precision for score in scores]) / len(scores):.2f}')
    print(f'rouge1 recall: {sum([score["rouge1"].recall for score in scores]) / len(scores):.2f}')
    print(f'rouge1 f-score: {sum([score["rouge1"].fmeasure for score in scores]) / len(scores):.2f}')
    print(f'rouge2 precision: {sum([score["rouge2"].precision for score in scores]) / len(scores):.2f}')
    print(f'rouge2 recall: {sum([score["rouge2"].recall for score in scores]) / len(scores):.2f}')
    print(f'rouge2 f-score: {sum([score["rouge2"].fmeasure for score in scores]) / len(scores):.2f}')
    print(f'rougeL precision: {sum([score["rougeL"].precision for score in scores]) / len(scores):.2f}')
    print(f'rougeL recall: {sum([score["rougeL"].recall for score in scores]) / len(scores):.2f}')
    print(f'rougeL f-score: {sum([score["rougeL"].fmeasure for score in scores]) / len(scores):.2f}')

def test_rouge_logger():
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot, Rouge_Logger
    from main import Sector_2024
    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)
    model_wrapper = ModelWrapper(Sector_2024())
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=30, logger=logger)

def test_normal_logger():
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot, Logger
    from main import Sector_2024
    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)
    model_wrapper = ModelWrapper(Sector_2024())
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=30, logger=logger)

def test_ablation():
    from ablation import Sector_without_crf, Sector_without_title, Sector_without_roberta
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot, Rouge_Logger
    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)
    # 分别测试Sector_without_crf, Sector_without_title, Sector_without_roberta
    # 测试Sector_without_crf
    model_wrapper = ModelWrapper(Sector_without_crf())
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=30, logger=logger)
    # 测试Sector_without_title
    model_wrapper = ModelWrapper(Sector_without_title())
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=30, logger=logger)
    # 测试Sector_without_roberta
    model_wrapper = ModelWrapper(Sector_without_roberta())
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=30, logger=logger)

def test_add_module():
    from main import Sector_2024
    from add_module import Sector_bert_vanilla, Sector_bert_crf_on, Sector_bert_title_on, Sector_roberta_vanilla
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot, Rouge_Logger
    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)
    # 测试Sector_bert_vanilla
    model_wrapper = ModelWrapper(Sector_bert_vanilla())
    assert model_wrapper.model.crf is None
    assert model_wrapper.model.tokenizer.name_or_path == 'bert-base-cased'
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=20, logger=logger)
    # 测试Sector_bert_crf_on
    model_wrapper = ModelWrapper(Sector_bert_crf_on())
    assert model_wrapper.model.crf is not None
    assert model_wrapper.model.tokenizer.name_or_path == 'bert-base-cased'
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=20, logger=logger)
    # 测试Sector_bert_title_on
    model_wrapper = ModelWrapper(Sector_bert_title_on())
    assert model_wrapper.model.crf is None
    assert model_wrapper.model.tokenizer.name_or_path == 'bert-base-cased'
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=20, logger=logger)
    # 测试Sector_roberta_vanilla
    model_wrapper = ModelWrapper(Sector_roberta_vanilla())
    assert model_wrapper.model.crf is None
    assert model_wrapper.model.tokenizer.name_or_path == 'roberta-base'
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=20, logger=logger)
    # 测试Sector_2024
    model_wrapper = ModelWrapper(Sector_2024())
    assert model_wrapper.model.crf is not None
    assert model_wrapper.model.tokenizer.name_or_path == 'roberta-base'
    model_wrapper.set_meta(fold_index=0, repeat_index=999)
    logger = Rouge_Logger(model_wrapper.get_name())
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=20, logger=logger)

def test_encode():
    import numpy as np
    from main import encode_with_title_and_sentences_truncate, encode_without_title, get_tokenizer
    tokenizer = get_tokenizer('roberta-base')
    sentences = ['This is a test sentence.', 'This is another test sentence.']
    title = 'This is a test title.'
    token_ids, head_ids = encode_with_title_and_sentences_truncate(tokenizer, sentences, title, empty_title=False)
    token_ids = np.array(token_ids)
    head_ids = np.array(head_ids)
    print(tokenizer.decode(token_ids))
    print(token_ids[token_ids])
    token_ids, head_ids = encode_with_title_and_sentences_truncate(tokenizer, sentences, title, empty_title=True)
    token_ids = np.array(token_ids)
    head_ids = np.array(head_ids)
    print(tokenizer.decode(token_ids))
    print(token_ids[token_ids])
    token_ids, head_ids = encode_without_title(tokenizer, sentences)
    print(tokenizer.decode(token_ids))
    token_ids = np.array(token_ids)
    print(token_ids[head_ids])

if __name__ == "__main__":  # {{ edit_2 }}
    main()  # {{ edit_3 }}