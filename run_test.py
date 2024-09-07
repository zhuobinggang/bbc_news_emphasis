def main():
    from dataset_verify import final_dataset_need_devset
    from trainer import ModelWrapper, train_and_plot
    from main import Sector_2024

    # 获取数据集
    train_set, dev_set, test_set = final_dataset_need_devset(shuffle=True, fold_index=0)

    # 初始化模型
    model = Sector_2024()

    # 包装模型
    model_wrapper = ModelWrapper(model)

    # 训练模型并打印结果
    train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=4, check_step=100, total_step=2000)


if __name__ == "__main__":  # {{ edit_2 }}
    main()  # {{ edit_3 }}