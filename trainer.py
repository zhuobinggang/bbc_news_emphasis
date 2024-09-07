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
        plt.plot(x[:len(y)], y, colors[i] if colors else None, label=l)
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
        for item in ds:
            results.extend(self.m.predict(item))
            labels.extend(item['labels'])
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

def train_and_plot(
        m, trainset, devset, testset, 
        batch_size = 4, check_step = 100, total_step = 2000): # 大约4个epoch
    # Init Model
    # Start
    import numpy as np
    trainset = Infinite_Dataset(trainset.copy())
    loss_plotter = Loss_Plotter()
    score_plotter = Score_Plotter()
    best_dev = 0
    for step in range(total_step):
        batch_loss = []
        for _ in range(batch_size):
            loss = m.loss(trainset.next())
            loss.backward()
            batch_loss.append(loss.item())
        loss_plotter.add(np.mean(batch_loss))
        m.opter_step()
        if (step + 1) % check_step == 0: # Evalue
            score_dev = m.test(devset.copy())
            score_test = m.test(testset.copy())
            beutiful_print_result(step, score_dev, score_test)
            if score_dev[2] > best_dev:
                best_dev = score_dev[2]
                # save_checkpoint(name, m, step, score_dev, score_test)
            # Plot & Cover
            score_plotter.add(score_dev[2], score_test[2])
            score_plotter.plot(f'logs/{m.get_name()}_score.png')
            loss_plotter.plot(f'logs/{m.get_name()}_loss.png')