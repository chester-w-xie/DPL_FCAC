import random
import torch
import os
import time

import numpy as np
import pprint as pprint

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def calculate_accuracy(target, predict, classes_num1, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num1)
    total = np.zeros(classes_num1)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num2):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num2, classes_num2))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D '
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h '
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm '
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's '
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms '
        i += 1
    if f == '':
        f = '0ms '
    return f


class CategoriesSamplerCEC():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls  # 采样的类别数
        self.n_per = n_per  # 每类采样的样本数

        label = np.array(label)  # all data label，数据集内全部样本的标签
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)  # - 把相同标签下的全部样本的索引取出并逐一堆叠，每类的索引单独用一个列表保存起来
            # 因此， self.m_ind中类别的个数等于数据集的总类别数

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            # 先随机采样类别镖旗
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            # 进一步逐个类别进行样本的采样
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            # 这里采样的每个batch内的数据是按照类别先后来排列的，
            yield batch


# CEC采样器的基础上，直接多采样一倍的数据用于真实数据的增量
class CategoriesSamplerCECV2():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls  # 采样的类别数
        self.n_per = n_per  # 每类采样的样本数

        label = np.array(label)  # all data label，数据集内全部样本的标签
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)  # - 把相同标签下的全部样本的索引取出并逐一堆叠，每类的索引单独用一个列表保存起来
            # 因此， self.m_ind中类别的个数等于数据集的总类别数

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            batch1 = []
            batch2 = []
            # 先随机采样类别标签，，直接翻倍采样，一半作为伪基类，另一半作为伪新类
            classes = torch.randperm(len(self.m_ind))[:(self.n_cls * 2)]  # random sample num_class indexs,e.g. 5
            classes1 = classes[:self.n_cls]
            classes2 = classes[self.n_cls:]
            # print(f'check 20, classes: {classes}')
            # print(f'check 21, classes1: {classes1}')
            # print(f'check 22, classes2: {classes2}')
            # 进一步逐个类别进行样本的采样
            for c1 in classes1:
                l = self.m_ind[c1]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch1.append(l[pos])
            batch1 = torch.stack(batch1).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            # 这里采样的每个batch内的数据是按照类别先后来排列的，
            for c2 in classes2:
                l = self.m_ind[c2]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch2.append(l[pos])
            batch2 = torch.stack(batch2).t().reshape(-1)
            batch = torch.cat([batch1, batch2], dim=0)
            # batch.cat(batch2)
            # print(f'check 23, batch1: {batch1}')
            # print(f'check 24, batch2: {batch2}')
            # print(f'check 25, batch: {batch}')
            yield batch


# 该采样器只在Session0 的时候被调用，因此它只是设计用来处理base类的数据，也就是执行meta-train的任务
class CategoriesSampler:

    def __init__(self, index, lenth, way, shot):
        self.index = index  # - 数据集中样本的索引，在数据集定义中已经按类别进行了归类
        # - 例如：打印dataset_train.sub_indexes[99]可以看到：array([97, 146,301,...])
        #  - 这里的样本索引号是在整个数据集内全局唯一的
        self.lenth = lenth  # - 按照默认args.batch_size加载原数据集所需要的迭代次数
        self.way = way
        self.shot = shot
        # print(f'index-1:{index[1]}')

    def __len__(self):
        return self.lenth

    def __iter__(self):
        # -逐一构建每个batch所需要的数据
        # - 再数据加载过程中，这里只被调用一次，即初始化的时候
        # 也即就是说，在初始化阶段，采样器已经把要采样的数据都以batch的方式打包好了
        # print(f'check- in iter')
        for lenth in range(self.lenth):  # - 逐一构建每个batch的数据（实际为数据的索引）
            batch = []
            labels_in_dataset = list(self.index.keys())  # - keys中的内容实际上就是数据集内的标签, 这里是获取数据集内的全部标签，以list返回
            random.shuffle(labels_in_dataset)  # - 把标签顺序打乱
            classes_for_fs = labels_in_dataset[:self.way]  # - 取出self.way个标签，结合上一步操作，就相当于每次迭代随机取出self.way类
            # 接着遍历选中的每一类，进行样本采集
            for c in classes_for_fs:
                data_indexes_in_one_class = torch.from_numpy(self.index[c])  # - 把c类的全部样本索引取出
                num_data_in_one_class = len(data_indexes_in_one_class)  # - 默认情况下，训练集内的每个类别都拥有500个样本
                # print(f'num_data_in_one_class:{num_data_in_one_class}')
                # 将0~num_data_in_one_class（包括0和num_data_in_one_class-1）随机打乱后获得一组数字序列并取self.shot个数字
                shot = torch.randperm(num_data_in_one_class)[:self.shot]
                shot_real = data_indexes_in_one_class[:self.shot]
                # 注意，这里获得的shot并不是样本的真实索引，e.g. [398,294,225,270,205] ,每个数字都不会大于num_data_in_one_class,默认500
                # print(f'shot:{shot}')
                # print(f'shot_real:{shot_real}')
                # print(f'C:{c}, c*num_data_in_one_class+shot:{c*num_data_in_one_class+shot}')
                batch.append((c * num_data_in_one_class + shot).int())  # - 将每n_shot个样本“索引”逐一添加到list中
                # 默认情况下，每类所拥有的样本数是一致的，因此，这里对shot中数字的增量区别仅在于类别编码c
                # print(f'batch in c:{c}----{batch}')
            # batch = torch.stack(batch).reshape(-1)
            # 先将batch内的多个索引列表进行堆叠成way * shot的一个索引矩阵，然后再将其拉平成为一个列表
            temp = torch.stack(batch)
            batch = temp.reshape(-1)
            # print(f'lenth:{lenth}, temp:{temp.shape}, batch in lenth:{batch.shape}')
            yield batch
            # 每次lenth 循环，到这里都会返回batch内的数据，然后跳出函数，等下次调用本函数时再接着执行下一个lenth 循环
            # 有个奇怪的地方是第一个调用本函数时，直接先返回了9个batch，但是再第一次迭代时没有用到，反而是留到了最后的9个batch
            # 另外，需要特别注意的是，这里返回的样本索引部是真实再数据集中的索引，而是一组相对的数字


# 单独应对不同初始基类的实验采样，这里进来的label还是原始的数字编码，因此
class CategoriesSamplerCECV3():

    def __init__(self, label, n_batch, n_cls, n_per, base_start_index):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls  # 采样的类别数
        self.n_per = n_per  # 每类采样的样本数
        print(f'check1 label in :{label}')
        print(f'check1 label in :{label-base_start_index}')
        label = label - base_start_index
        label = np.array(label)  # all data label，数据集内全部样本的标签
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)  # - 把相同标签下的全部样本的索引取出并逐一堆叠，每类的索引单独用一个列表保存起来
            # 因此， self.m_ind中类别的个数等于数据集的总类别数

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            # 先随机采样类别标签
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            # 进一步逐个类别进行样本的采样
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            # 这里采样的每个batch内的数据是按照类别先后来排列的，
            yield batch

