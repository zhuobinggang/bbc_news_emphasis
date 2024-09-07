import numpy as np

def cal_prec_rec_f1_v2(results, targets):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for guess, target in zip(results, targets):
        if guess == 1:
            if target == 1:
                TP += 1
            elif target == 0:
                FP += 1
        elif guess == 0:
            if target == 1:
                FN += 1
            elif target == 0:
                TN += 1
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
    balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
    return prec, rec, f1, balanced_acc


def draw_line_chart(x, ys, legends, path='./logs/dd.png', colors=None, xlabel=None, ylabel=None):
    import matplotlib.pyplot as plt
    plt.clf()
    for i, (y, l) in enumerate(zip(ys, legends)):
        if colors:
            plt.plot(x[:len(y)], y, colors[i], label=l)
        else:
            plt.plot(x[:len(y)], y, label=l)
    plt.legend()
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.savefig(path)

def beutiful_print_result(step, dev_res, test_res):
    dev_prec, dev_rec, dev_f, _ = dev_res
    test_prec, test_rec, test_f, _ = test_res
    print(f'STEP: {step + 1}\nDEV: {round(dev_f, 5)}\nTEST: {round(test_f, 5)}\n\n')

class ModelWrapper:
    def __init__(self, m):
        self.m = m
        self.model = m
    def loss(self, item):
        return self.m.get_loss(item)
    def opter_step(self):
        self.m.opter.step()
        self.m.opter.zero_grad()
    def get_name(self):
        return self.m.get_name()
    def test(self, ds):
        labels = []
        results = []
        for index, item in enumerate(ds):  # 添加下标
            try:
                results.extend(self.m.predict(item))
                labels.extend(item['labels'])
            except RuntimeError as e:
                print(f'Error at index {index}: {item}')  # 打印下标和item
                raise e  # 继续抛出错误以终止程序运行
        return cal_prec_rec_f1_v2(results, labels)
        

class Infinite_Dataset:
    def __init__(self, ds):
        self.ds = ds
        self.counter = 0

    def next(self):
        item = self.ds[self.counter]
        self.counter = (self.counter + 1) % len(self.ds)  # 精简计数器逻辑
        return item

class Loss_Plotter():
    def __init__(self):
        self.lst = []

    def add(self, item):
        self.lst.append(item)

    def plot(self, path, name = 'loss'):
        draw_line_chart(range(len(self.lst)), [self.lst], [name], path = path)

class Score_Plotter():
    def __init__(self):
        self.l1 = []
        self.l2 = []

    def add(self, item1, item2):
        self.l1.append(item1)
        self.l2.append(item2)

    def plot(self, path, name1 = 'dev', name2 = 'test'):
        draw_line_chart(range(len(self.l1)), [self.l1, self.l2], [name1, name2], path = path)

def fake_test_score():
    return -1.0, -1.0, -1.0, -1.0

class Logger:
    def __init__(self, model_name):
        self.loss_plotter = Loss_Plotter()
        self.score_plotter = Score_Plotter()
        self.batch_log = {'loss': []}
        self.checkpoint_log = {'scores': [], 'log_time': []}
        self.txt_log = ''
        self.model_name = model_name
        self.best_dev = 0
        self.best_test = 0
    def start(self):
        import time  # 导入时间模块
        self.start_time = time.time()  # 记录开始时间
    def log_batch(self, batch_loss):
        mean_loss = np.mean(batch_loss)
        self.batch_log['loss'].append(mean_loss)
        self.loss_plotter.add(mean_loss)
    def log_checkpoint(self, dev_score):
        import time  # 导入时间模块
        meta = {'best_score_updated': False}
        self.checkpoint_log['scores'].append({'dev': dev_score, 'test': fake_test_score()})
        self.checkpoint_log['log_time'].append(time.time())
        self.score_plotter.add(dev_score[2], self.best_test)
        if dev_score[2] > self.best_dev:
            self.best_dev = dev_score[2]
            meta['best_score_updated'] = True
        return meta
    def update_best_test(self, test_score):
        self.best_test = test_score[2]
        self.checkpoint_log['scores'][-1]['test'] = test_score
    def end(self):
        import time  # 导入时间模块
        self.end_time = time.time()  # 记录结束时间
        self.score_plotter.plot(f'logs/{self.model_name}_score.png')
        self.loss_plotter.plot(f'logs/{self.model_name}_loss.png')
        self.log_txt()
    def log_txt(self):
        import time  # 导入时间模块
        best_dev = self.best_dev
        best_test = self.best_test
        with open(f'logs/{self.model_name}_log.txt', 'w') as f:
            # 输出批次损失
            # f.write(f'Batch loss: {self.batch_log["loss"]}\n\n')
            total_time = self.end_time - self.start_time  # 计算总时间
            f.write(f'total time: {total_time:.2f}s\n')
            f.write(f'best dev: {best_dev}\n')
            f.write(f'best test: {best_test}\n\n')
            for i, score in enumerate(self.checkpoint_log['scores']):
                log_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(self.checkpoint_log['log_time'][i]))
                f.write(f'# Check point {i}\n')
                f.write(f'log time: {log_time}\n')
                f.write('## dev score\n')
                f.write(f'precision: {score["dev"][0]:.2f}\n')
                f.write(f'recall: {score["dev"][1]:.2f}\n')
                f.write(f'f-score: {score["dev"][2]:.2f}\n')
                f.write('## test score\n')
                f.write(f'precision: {score["test"][0]:.2f}\n')
                f.write(f'recall: {score["test"][1]:.2f}\n')
                f.write(f'f-score: {score["test"][2]:.2f}\n\n')
            
def save_checkpoint(model):
    import torch
    PATH = f'/usr01/taku/checkpoint/bbc_news_emphasis/{model.get_name()}.checkpoint'
    torch.save({
            'model_state_dict': model.m.state_dict(),
            }, PATH)

def train_and_plot(
        m, trainset, devset, testset, 
        batch_size = 4, check_step = 100, total_step = 2000): # 大约4个epoch
    # Init Model
    # Start
    trainset = Infinite_Dataset(trainset.copy())
    logger = Logger(m.get_name())
    logger.start()
    for step in range(total_step):
        batch_loss = []
        for _ in range(batch_size):
            loss = m.loss(trainset.next())
            loss.backward()
            batch_loss.append(loss.item())
        logger.log_batch(batch_loss)
        m.opter_step()
        if (step + 1) % check_step == 0: # Evalue
            score_dev = m.test(devset.copy())
            # Plot & Cover
            meta = logger.log_checkpoint(score_dev)
            if meta['best_score_updated']:
                score_test = m.test(testset.copy())
                logger.update_best_test(score_test)
                save_checkpoint(m)
                print('Best score updated!')
            beutiful_print_result(step, score_dev, score_test)
    logger.end()