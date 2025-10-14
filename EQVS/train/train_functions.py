"""
DATE: 10/09/2021
LAST CHANGE: 30/11/2021
AUTHOR: CHENG ZHANG

training functions
"""
import torch
import os
import shutil
import time
import copy
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import csv

from model import model
from train import utils_torch


def cal_metrices(outputs, labels):
    """
    Calculate accuracy with the output of network and real label
    :param outputs: the output of the network
    :param labels: the real label of the data
    :return: accuracy, precision, recall, false positive rate
    """
    # (4, 1000, 15) --> (4, 250, 5)
    # slice_r = 4
    # slice_c = 3
    outputs = torch.round(outputs)

    prediction = outputs
    # one_hot_prediction = convert_one_hot(argmax.view(-1, 1).float())
    batch, label_predict = outputs.size()
    # label_count = batch * frame * codeword
    accuracy = (labels == prediction).float().mean()
    zero_labels = torch.tensor(np.zeros((batch, label_predict))).cuda()
    one_labels = torch.tensor(np.ones((batch, label_predict))).cuda()
    outputs = torch.round(outputs)

    # False positive, original 0, prediction 1
    fp = ((labels == zero_labels) & (outputs == one_labels)).float().sum()
    # False negative, original 1, prediction 0
    fn = ((labels == one_labels) & (outputs == zero_labels)).float().sum()
    # true positive, original 1, prediction 1
    tp = ((labels == one_labels) & (outputs == one_labels)).float().sum()
    # true negative, original 0, prediction 0
    tn = ((labels == zero_labels) & (outputs == zero_labels)).float().sum()

    if tp + fp == 0.:
        precision = torch.tensor(0.)
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0.:
        recall = torch.tensor(0.)
    else:
        recall = tp / (tp + fn)
    # false positive rate
    if tn + fp == 0.:
        fp_rate = torch.tensor(0.)
    else:
        fp_rate = fp / (tn + fp)

    return accuracy, precision, recall, fp_rate


def get_metrics(outputs, labels):
    """
    Calculate accuracy with the output of network and real label
    :param outputs: the output of the network
    :param labels: the real label of the data
    :return: accuracy, precision, recall, false positive rate
    """
    # (4, 1000, 15) --> (4, 250, 5)
    # slice_r = 4
    # slice_c = 3
    outputs = torch.round(outputs)

    prediction = outputs
    # one_hot_prediction = convert_one_hot(argmax.view(-1, 1).float())
    batch, label_predict = outputs.size()
    # label_count = batch * frame * codeword
    accuracy = (labels == prediction).float().mean()
    zero_labels = torch.tensor(np.zeros((batch, label_predict))).cuda()
    one_labels = torch.tensor(np.ones((batch, label_predict))).cuda()
    outputs = torch.round(outputs)

    # False positive, original 0, prediction 1
    fp = ((labels == zero_labels) & (outputs == one_labels)).float().sum()
    # False negative, original 1, prediction 0
    fn = ((labels == one_labels) & (outputs == zero_labels)).float().sum()
    # true positive, original 1, prediction 1
    tp = ((labels == one_labels) & (outputs == one_labels)).float().sum()
    # true negative, original 0, prediction 0
    tn = ((labels == zero_labels) & (outputs == zero_labels)).float().sum()

    return tp, tn, fp, fn


# def get_path(embed_rate):
#     """
#     Return data path and number of workers(used in data loader) depends on operating system
#     :return: cover_dir, stego_dir, num_workers
#     """
#     cover_dir = "../../../data/G729CNV_10.0S/code_em0.0"
#     stego_dir = "../../../data/G729CNV_10.0S/code_em" + embed_rate
#     num_workers = 6
#     return cover_dir, stego_dir, num_workers

def get_path(embed_rate):
    """
    Return data path and number of workers(used in data loader) depends on operating system
    :return: cover_dir, stego_dir, num_workers
    """
    # cover_dir = "../../../data/G729CNV_10.0S/code_em0.0"
    # stego_dir = "../../../data/G729CNV_10.0S/code_em" + embed_rate
    # cover_dir = "C:/博士/毕业论文/data/G729CNV_10.0S/code_em0.0"
    # stego_dir = "C:/博士/毕业论文/data/G729CNV_10.0S/code_em" + embed_rate
    cover_dir = "/home/cheng/Steganalysis/data/G729CNV_10.0S/code_em0.0"
    stego_dir = "/home/cheng/Steganalysis/data/G729CNV_10.0S/code_em" + embed_rate

    num_workers = 6
    return cover_dir, stego_dir, num_workers

###############################################################################
# The following functions are used for retaining history training information #
###############################################################################


def get_info_path(epoch, save_dir, mode):
    """
    :param epoch: current epoch number
    :param save_dir: target saving dir
    :param mode: 'best' or 'latest'
    :return: recording paths
    """
    if mode == 'best':
        model_file = save_dir + 'model_best_' + str(epoch) + '.pth'
    elif mode == 'cal':
        model_file = save_dir + 'model_cal' + str(epoch) + '.pth'
    else:
        model_file = save_dir + 'model_latest_' + str(epoch) + '.pth'
    epoch_record_file = save_dir + 'epoch_record.txt'
    acc_record_file = save_dir + 'accuracy_record.txt'
    pre_record_file = save_dir + 'precision_record.txt'
    return model_file, epoch_record_file, acc_record_file, pre_record_file


def save_model(state, epoch, save_dir, mode, metric_value):
    """
    Remove the model_best dir if exist and store the current best epoch
    :param state: current model state
    :param epoch: current epoch number
    :param save_dir: target saving dir
    :param mode: either 'best', 'latest' or 'query'
    :param metric_value: corresponding accuracy or precision value
    """
    # determine save_dir
    model_file, epoch_record_file, acc_record_file, pre_record_file = get_info_path(epoch, save_dir, mode)
    (accuracy, precision) = metric_value

    # save model
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
    torch.save(state, model_file)
    if epoch is not None:
        epoch = int(epoch)
    if accuracy is not None:
        accuracy = float(accuracy)
    if precision is not None:
        precision = float(precision)
    check_save(epoch_record_file, epoch)
    check_save(acc_record_file, accuracy)
    check_save(pre_record_file, precision)


def check_save(path, value):
    """
    Save the value unless its None
    :param path: save path
    :param value: value to save
    """
    if value is None:
        pass
    else:
        f = open(path, 'w')
        f.write(str(value))
        f.flush()
        f.close()


def load_states(best_acc_dir, best_pre_dir, latest_dir):
    """
    Load history training state
    :param best_acc_dir: the dir that save the best accuracy model
    :param best_pre_dir: the dir that save the best precision model
    :param latest_dir: the dir that save the latest model
    :return: best accuracy(0. if no history training), best precision(0. if no history training),
                latest epoch(0 if no history training)
    """
    best_acc = load_value(best_acc_dir, 'accuracy')
    best_pre = load_value(best_pre_dir, 'precision')
    latest_epoch = int(load_value(latest_dir, 'epoch'))

    return best_acc, best_pre, latest_epoch


def load_value(dir, target):
    """
    Check whether the dir exists. Load the corresponding value if exist
    :param dir: dir for search
    :param target: target to load. 'epoch', 'accuracy' or 'precision'
    :return: target value
    """
    load_val = 0.
    # check whether dir exist
    if not os.path.exists(dir):
        pass
    else:
        files = os.listdir(dir)
        load_val = check_file_load(dir, files, target)
    return load_val


def check_file_load(dir, files, target):
    """
    Inner loop for load_value, check whether the file exists. Load the corresponding value if exist
    :param files: files for search
    :param target: target to load. 'epoch', 'accuracy' or 'precision'
    :return: target value
    """
    for file in files:
        if target in file:
            f = open(dir + file, 'r')
            if target == 'epoch':
                value = int(f.readline())
            else:
                value = float(f.readline())
            return value
    return 0.


def load_latest_model(latest_dir):
    files = os.listdir(latest_dir)
    for file in files:
        if '.pth' in file:
            model_path = latest_dir + file
            return torch.load(model_path)
    print('Warning, can\'t find the saved latest model')
    return None


def clean_file(record_file):
    reader = open(record_file, 'r')
    to_write_temp = []
    to_write_checked = []
    for line in reader.readlines():
        to_write_temp.append(line)
        if line.strip('\n') == 'Saving complete':
            to_write_checked = copy.deepcopy(to_write_temp)
    reader.close()
    writer = open(record_file, 'w')
    writer.writelines(to_write_checked)
    writer.flush()
    writer.close()
    print('finish cleaning the previous txt-record file')


#########################################################
# The following functions are used for training process #
#########################################################


def single_train(train_loader, f, epoch, net, loss_func, optimizer):
    """
    Training the network using all the data, feed the data into previous network if current FAE is not
    the first block in the architecture
    """
    net.train()
    train_loss = 0.
    train_accuracy = 0.
    train_precision = 0.
    train_recall = 0.
    train_fp_rate = 0.
    for batch_idx, data in enumerate(train_loader):
        # sys.stdout.write('\rCurrent batch index: %s' % batch_idx)
        # sys.stdout.flush()
        # last three dim, used for CNV
        X, T = Variable(data['data'].cuda()).long(), Variable(data['label'].cuda()).long()

        output = net(X)

        optimizer.zero_grad()
        current_loss = loss_func(output.float(), T.float())
        current_loss.backward()
        optimizer.step()
        current_accuracy, current_precision, current_recall, current_fp_rate = cal_metrices(output, T)

        train_loss += current_loss.item()
        train_accuracy += current_accuracy.item()
        train_precision += current_precision.item()
        train_recall += current_recall.item()
        train_fp_rate += current_fp_rate.item()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    train_precision /= len(train_loader)
    train_recall /= len(train_loader)
    train_fp_rate /= len(train_loader)
    print('\nTrain epoch: ' + str(epoch))
    print('Epoch_train_loss: ' + str(train_loss))
    print('Epoch_train_accuracy: ' + str(train_accuracy))
    print('Epoch_train_precision: ' + str(train_precision))
    print('Epoch_train_recall: ' + str(train_recall))
    print('Epoch_train_fp_rate: ' + str(train_fp_rate))
    f.write('\nTrain epoch: ' + str(epoch) + '\n')
    f.write('Epoch_train_loss: ' + str(train_loss) + '\n')
    f.write('Epoch_train_accuracy: ' + str(train_accuracy) + '\n')
    f.write('Epoch_train_precision: ' + str(train_precision) + '\n')
    f.write('Epoch_train_recall: ' + str(train_recall) + '\n')
    f.write('Epoch_train_fp_rate: ' + str(train_fp_rate) + '\n')
    f.flush()

def single_eval(test_loader, f, epoch, net, loss_func):
    """
    Training the network using all the data, feed the data into previous network if current FAE is not
    the first block in the architecture
    """
    net.eval()
    tp_all = 0.
    tn_all = 0.
    fp_all = 0.
    fn_all = 0.
    for batch_idx, data in enumerate(test_loader):
        # sys.stdout.write('\rCurrent batch index: %s' % batch_idx)
        # sys.stdout.flush()
        # last three dim, used for CNV
        X, T = Variable(data['data'].cuda()).long(), Variable(data['label'].cuda()).long()

        output = net(X)

        current_loss = loss_func(output.float(), T.float())
        # TODO, summarize and record precision,recall,fp_rate
        tp, tn, fp, fn = get_metrics(output, T)
        tp_all += tp
        tn_all += tn
        fp_all += fp
        fn_all += fn

    if tp_all + fp_all == 0.:
        precise_test_precision = torch.tensor(0.)
    else:
        precise_test_precision = tp_all / (tp_all + fp_all)
    if tp_all + fn_all == 0.:
        precise_test_recall = torch.tensor(0.)
    else:
        precise_test_recall = tp_all / (tp_all + fn_all)
    # false positive rate
    if tn_all + fp_all == 0.:
        precise_test_fp_rate = torch.tensor(0.)
    else:
        precise_test_fp_rate = fp_all / (tn_all + fp_all)

    precise_test_accuracy = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all)

    # evaluating time
    print('Precise evaluation metrics')
    f.write('\nPrecise evaluation metrics')
    # accuracy
    print('Epoch: ' + str(epoch) + ', precise acc: ' + str(precise_test_accuracy))
    f.write('\nEpoch: ' + str(epoch) + ', precise acc: ' + str(precise_test_accuracy))
    # precision
    print('Epoch: ' + str(epoch) + ', precise precision: ' + str(precise_test_precision))
    f.write('\nEpoch: ' + str(epoch) + ', precise precision:' + str(precise_test_precision))
    # recall
    print('Epoch: ' + str(epoch) + ', precise recall: ' + str(precise_test_recall))
    f.write('\nEpoch: ' + str(epoch) + ', precise recall: ' + str(precise_test_recall))
    # false positive
    print('Epoch: ' + str(epoch) + ', precise false positive rate: ' + str(precise_test_fp_rate))
    f.write('\nEpoch: ' + str(epoch) + ', precise false positive rate: ' + str(precise_test_fp_rate))

    return precise_test_precision, precise_test_accuracy


def full_train(language, embed_rate, sample_length, epochs, batch_size, pretrain):
    """
    This function runs a single fast auto-encoder network
    """
    torch.set_float32_matmul_precision('highest')
    # set output, best precision and accuracy record path
    sample_length_float = sample_length
    sample_length = str(sample_length) + "S"
    result_root = "./train/results/"
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    record_path = result_root + "729CNV_" + language + "_" + str(embed_rate) + "_" + sample_length + "_" + ".txt"
    save_root = result_root + "729CNV_" + str(embed_rate) + "_" + sample_length
    best_acc_dir = save_root + "_best_accuracy_" + language + "/"
    best_pre_dir = save_root + "_best_precision_" + language + "/"
    latest_dir = save_root + "_" + "latest" + "_" + language + "/"

    # load history information, use default information if history information not exist
    best_acc, best_pre, latest_epoch = load_states(best_acc_dir, best_pre_dir, latest_dir)

    # initialize data loader
    print('Initializing data loader')
    batch_size = batch_size
    pin_memory = True
    cover_dir, stego_dir, num_workers = get_path(str(embed_rate))
    train_loader, valid_loader, test_loader = utils_torch.read_data(cover_dir, stego_dir, batch_size, num_workers,
                                                                    pin_memory, language, sample_length_float)
    print('Data loader initialized')

    net = model.Model(sample_length_float).cuda()

    write_mode = 'w'
    # check whether the previous trained model exist or not
    if latest_epoch != 0:
        # exist previous trained model, clean the previous record file
        write_mode = 'a+'
        state_dict = load_latest_model(latest_dir)
        print('History training found, loading the latest trained model')
        net.load_state_dict(state_dict)
        print('The latest trained model has been loaded')
        clean_file(record_path)
        epochs = epochs - latest_epoch
        print('Epochs left:' + str(epochs))
    loss_function = nn.BCELoss().cuda()
    # weight_pos = 0.8
    # weight_neg = 1-weight_pos
    # loss_function = weighted_BCEloss.CustomBCELoss(weight_pos*2, weight_neg*2)
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': 1e-4}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=latest_epoch)

    best_accuracy = best_acc
    best_precision = best_pre
    f = open(record_path, write_mode)
    for epoch in range(epochs):
        epoch_count = epoch + 1 + latest_epoch
        single_train(train_loader, f, epoch_count, net, loss_function, optimizer)
        current_precision, current_accuracy = single_eval(valid_loader, f, epoch_count, net, loss_function)
        tested = False
        if current_precision >= best_precision:
            best_precision = current_precision
            save_model(net.state_dict(), epoch_count, best_pre_dir, 'best', (None, best_precision))
            print('Best precision epoch: ' + str(epoch_count))
            f.write('Best precision epoch: ' + str(epoch_count) + '\n')
            print('Evaluating on test set')
            single_eval(test_loader, f, epoch_count, net, loss_function)
            tested = True
            f.flush()
        if current_accuracy >= best_accuracy:
            best_accuracy = current_accuracy
            save_model(net.state_dict(), epoch_count, best_acc_dir, 'best', (best_accuracy, None))
            print('Best accuracy epoch: ' + str(epoch_count))
            f.write('Best accuracy epoch: ' + str(epoch_count) + '\n')
            if tested is False:
                print('Evaluating on test set')
                single_eval(test_loader, f, epoch_count, net, loss_function)
            f.flush()
        # Save each latest epoch
        save_model(net.state_dict(), epoch_count, latest_dir, 'latest', (None, None))
        scheduler.step()
        f.write('Saving complete' + '\n')
        f.flush()
    f.close()


def single_speed_test(test_loader, net):
    """
    Training the network using all the data, feed the data into previous network if current FAE is not
    the first block in the architecture
    """
    net.eval()
    start = time.time()
    for batch_idx, data in enumerate(test_loader):
        X, T = Variable(data['data'].cuda()).long(), Variable(data['label'].cuda()).long()
        output = net(X)
    end = time.time()

    return (end - start) / 5000


def speed_test(language, embed_rate, sample_length, batch_size):
    sample_length_float = sample_length
    sample_length = str(sample_length) + "S"
    result_root = "./train/results/"
    save_root = result_root + "729CNV_" + str(embed_rate) + "_" + sample_length
    best_acc_dir = save_root + "_best_accuracy_" + language + "/"
    speed_root = "./train/results/speed_test/"

    cover_dir, stego_dir, num_workers = get_path(str(embed_rate))

    # initialize data loader
    print('Initializing data loader')
    batch_size = batch_size
    pin_memory = True
    test_loader = utils_torch.speed_test_read_data(cover_dir, stego_dir, batch_size, num_workers, pin_memory,
                                                   language, sample_length_float)

    if not os.path.exists(speed_root):
        os.makedirs(speed_root)
    speed_path = speed_root + "729CNV_" + language + "_" + str(embed_rate) + "_" + sample_length + "_" + ".txt"

    net = model.Model(sample_length_float).cuda()

    state_dict = load_latest_model(best_acc_dir)
  
    print('History training found, loading the latest trained model')
    # net.load_state_dict(state_dict)
    print('The latest trained model has been loaded')

    cumulate_time = 0
    for i in range(10):
        test_time = single_speed_test(test_loader, net)
        cumulate_time += test_time
    average_time = cumulate_time / 10
    write_mode = 'w'
    f = open(speed_path, write_mode)
    print('Average speed: ' + str(average_time))
    f.write('Average speed: ' + str(average_time))
    f.close()


def get_inference_value(language, embed_rate, sample_length, batch_size):
    """
    This function runs a single fast auto-encoder network
    """
    torch.set_float32_matmul_precision('highest')
    # set output, best precision and accuracy record path
    sample_length_float = sample_length
    sample_length = str(sample_length) + "S"
    result_root = "./train/results/"
    if not os.path.exists(result_root):
        print('No trained model found')
        exit(0)
    record_root = (result_root + "Inference_values/")
    if not os.path.exists(record_root):
        os.makedirs(record_root)
    record_path = (record_root + "729CNV_" + language + "_" + str(embed_rate) + "_" + sample_length
                   + ".csv")
    save_root = result_root + "729CNV_" + str(embed_rate) + "_" + sample_length
    best_acc_dir = save_root + "_best_accuracy_" + language + "/"
    best_pre_dir = save_root + "_best_precision_" + language + "/"
    latest_dir = save_root + "_" + "latest" + "_" + language + "/"

    # load history information, use default information if history information not exist
    best_acc, best_pre, latest_epoch = load_states(best_acc_dir, best_pre_dir, latest_dir)

    # initialize data loader
    print('Initializing data loader')
    batch_size = batch_size
    pin_memory = True
    cover_dir, stego_dir, num_workers = get_path(str(embed_rate))
    data_loader = utils_torch.get_inference_value_read_data(cover_dir, stego_dir, batch_size, num_workers, pin_memory,
                                                      language, sample_length_float)
    print('Data loader initialized')

    # initialize model
    net = model.Model(sample_length_float).cuda()

    # check whether the previous trained model exist or not
    if latest_epoch != 0:
        # exist previous trained model, clean the previous record file
        write_mode = 'a+'
        state_dict = load_latest_model(latest_dir)
        print('History training found, loading the latest trained model')
        net.load_state_dict(state_dict)
        print('The latest trained model has been loaded')
    else:
        print('No trained model found')
        exit(0)

    f = open(record_path, 'w')
    head = ['Inference_value (x)', 'y (1-x)', 'True Label', 'Correct']
    writer = csv.writer(f)
    writer.writerow(head)
    get_inference_value_eval(data_loader, writer, net, batch_size)


def get_inference_value_eval(eval_loader, writer, net, batch_size):
    """
    Training the network using all the data, feed the data into previous network if current FAE is not
    the first block in the architecture
    """
    net.eval()
    current = 0
    total_num = len(eval_loader)
    for batch_idx, data in enumerate(eval_loader):
        # sys.stdout.write('\rCurrent batch index: %s' % batch_idx)
        # sys.stdout.flush()
        # last three dim, used for CNV
        X, T = Variable(data['data'].cuda()).long(), Variable(data['label'].cuda()).long()

        output = net(X)
        correct_num = 0
        prediction = torch.round(output)
        correct = (prediction == T).item()
        if correct:
            correct_num = 1

        x = output.view(batch_size).item()
        y = (1-output).view(batch_size).item()
        T = T.view(batch_size).item()

        current += 1
        print('\nSample Number: ' + str(current) + ' | Total Number: ' + str(total_num))

        to_write = [x, y, T, correct_num]
        writer.writerow(to_write)
        # TODO, summarize and record precision,recall,fp_rate
        current_accuracy, current_precision, current_recall, current_fp_rate = cal_metrices(output, T)
