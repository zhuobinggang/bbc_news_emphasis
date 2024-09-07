def main():
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot
    from main import Sector_2024

    for fold_index in range(5):
        for repeat_index in range(3):
            # 获取数据集
            train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=fold_index)
            # 初始化模型
            model = Sector_2024()
            model.set_meta(fold_index=fold_index, repeat_index=repeat_index)
            print(model.get_name())
            # 包装模型
            model_wrapper = ModelWrapper(model)
            # 训练模型并打印结果
            train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=40, total_step=400)


if __name__ == "__main__":  # {{ edit_2 }}
    main()  # {{ edit_3 }}
    import os
    os.system('shutdown -s')