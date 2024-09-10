def run_with_model_class(model_classes):
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot, Rouge_Logger

    for fold_index in range(5):
        # 获取数据集
        train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=fold_index)
        for repeat_index in range(3):
            for model_class in model_classes:
                # 初始化模型
                model_wrapper = ModelWrapper(model_class())
                model_wrapper.set_meta(fold_index=fold_index, repeat_index=repeat_index)
                print(model_wrapper.get_name())
                logger = Rouge_Logger(model_wrapper.get_name())  # 添加日志记录器
                # 训练模型并打印结果
                train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=40, total_step=400, logger=logger)  # 添加logger参数

def run_all():
    # 设置随机种子
    import numpy as np
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    # 运行所有模型
    from main import Sector_2024
    from ablation import Sector_without_roberta, Sector_without_crf, Sector_without_title
    run_with_model_class([Sector_2024, Sector_without_roberta, Sector_without_crf, Sector_without_title])

if __name__ == "__main__":  
    run_all() 
    import os
    os.system('shutdown')