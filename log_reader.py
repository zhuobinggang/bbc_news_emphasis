def read_log():
    prefixs = ["bert-base-title-on-crf-on", 
                       "roberta-base-title-off-crf-on", 
                       "roberta-base-title-on-crf-off", 
                       "roberta-base-title-on-crf-on"]
    for prefix in prefixs:
        for fold_index in range(5):
            for repeat_index in range(3):
                with open(f"logs/{prefix}-fold{fold_index}-repeat{repeat_index}_log.txt", "r") as file:
                    while True:
                        line = file.readline()
                        if not line:
                            break
                        if "best test (rouge1 f-score)" in line:
                            score = line.split("best test (rouge1 f-score): ")[1]
                            print(f"{prefix}-fold{fold_index}-repeat{repeat_index}: {score}")
                            break

def read_log_print_markdown_table(classes):
    prefixs = [cls.__name__ for cls in classes]
    print("| prefix | fold1 | fold2 | fold3 | fold4 | fold5 | average |")
    print("| -------- | -------- | -------- | -------- | -------- | -------- | -------- |")
    for prefix in prefixs:
        scores = [[], [], [], [], []]  # 用于存储每个fold的分数
        for fold_index in range(5):
            for repeat_index in range(3):
                with open(f"logs/{prefix}-fold{fold_index}-repeat{repeat_index}_log.txt", "r") as file:
                    while True:
                        line = file.readline()
                        if not line:
                            break
                        if "best test (rouge1 f-score)" in line:
                            score = float(line.split("best test (rouge1 f-score): ")[1])
                            scores[fold_index].append(score)  # 将分数添加到对应的fold列表中
                            break
        
        # 计算每个fold的平均分数
        avg_scores = [sum(fold_scores) / len(fold_scores) if fold_scores else 0 for fold_scores in scores]
        overall_average = sum(avg_scores) / len(avg_scores)  # 计算所有fold的平均值
        print(f"| {prefix} | {' | '.join(f'{avg:.3f}' for avg in avg_scores)} | {overall_average:.3f} |")

def read_log_print_markdown_table_ablation():
    from main import Sector_2024
    from ablation import Sector_without_roberta, Sector_without_crf, Sector_without_title
    classes = [Sector_2024, Sector_without_roberta, Sector_without_crf, Sector_without_title]
    read_log_print_markdown_table(classes)

def read_log_print_markdown_table_add_modules():
    from main import Sector_2024
    from add_module import Sector_bert_vanilla, Sector_bert_crf_on, Sector_bert_title_on, Sector_roberta_vanilla
    classes = [Sector_bert_vanilla, Sector_bert_crf_on, Sector_bert_title_on, Sector_roberta_vanilla, Sector_2024]
    read_log_print_markdown_table(classes)

if __name__ == "__main__":
    read_log_print_markdown_table_add_modules()
