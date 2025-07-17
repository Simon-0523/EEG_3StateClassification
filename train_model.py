
from utils import *
import copy
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from eeg_stft import compute_stft

CUDA = torch.cuda.is_available()


def normalize_1d(data):
    """
    this function do standard normalization for EEG channel by channel
    :param train: training data
    :param test: testing data
    :return: normalized training and testing data
    """
    # data: sample x 1 x channel x data
    mean = 0
    std = 0
    for channel in range(data.shape[2]):
        mean = np.mean(data[:, :, channel, :])
        std = np.std(data[:, :, channel, :])
        data[:, :, channel, :] = (data[:, :, channel, :] - mean) / std
        data[:, :, channel, :] = (data[:, :, channel, :] - mean) / std
    return data


def normalize_2d(data):
    """
    this function do standard normalization for EEG channel by channel
    :param train: training data
    :param test: testing data
    :return: normalized training and testing data
    """
    # data: sample x 1 x channel x data
    mean = 0
    std = 0
    for channel in range(data.shape[1]):
        mean = np.mean(data[:, channel, :, :])
        std = np.std(data[:, channel, :, :])
        data[:, channel, :, :] = (data[:, channel, :, :] - mean) / std
        data[:, channel, :, :] = (data[:, channel, :, :] - mean) / std
    return data


def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    loop = tqdm(data_loader, desc=Colors.GREEN + "Training batches" + Colors.RESET,
                total=len(data_loader), leave=True, ncols=100, ascii=False, colour='GREEN', unit='batch')
    for i, (x_batch, y_batch) in enumerate(loop):
        size_0 = x_batch.size(0)
        size_1 = x_batch.size(1)
        size_2 = x_batch.size(2)
        size_3 = x_batch.size(3)
        x_batch = x_batch.numpy()
        _, _, y1 = compute_stft(x_batch)  # y1[32,1,28,257,33]
        y1 = y1[:, :, :, 1:, :]
        y1 = y1.reshape(size_0*size_2, 1, 128, 8)
        y1 = normalize_2d(y1)
        y1 = torch.abs(torch.tensor(y1)).float()
        # y1 = torch.tensor(y1)
        y1 = y1.transpose(2, 3)
        y1 = y1.reshape(size_0, size_1, size_2, size_3)
        # y2 = x_batch.reshape(x_batch.size(0)*x_batch.size(2),10,256)
        x_batch = torch.from_numpy(normalize_1d(x_batch)).float()
        if CUDA:
            y1, x_batch, y_batch = y1.cuda(), x_batch.cuda(), y_batch.cuda()

        out = net(y1, x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
        loop.set_postfix(loss=loss.item())
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        loop = tqdm(data_loader, desc=Colors.YELLOW + "Validate batches" + Colors.RESET,
                    total=len(data_loader), leave=True, ncols=100, ascii=False, colour='YELLOW', unit='batch')
        for i, (x_batch, y_batch) in enumerate(loop):
            size_0 = x_batch.size(0)
            size_1 = x_batch.size(1)
            size_2 = x_batch.size(2)
            size_3 = x_batch.size(3)
            x_batch = x_batch.numpy()
            _, _, y1 = compute_stft(x_batch)  # y1[32,1,28,257,33]
            y1 = y1[:, :, :, 1:, :]
            y1 = y1.reshape(size_0*size_2, 1, 128, 8)
            y1 = normalize_2d(y1)
            y1 = torch.abs(torch.tensor(y1)).float()
            # y1 = torch.tensor(y1)
            y1 = y1.transpose(2, 3)
            y1 = y1.reshape(size_0, size_1, size_2, size_3)
            # y2 = x_batch.reshape(x_batch.size(0)*x_batch.size(2),10,256)
            x_batch = torch.from_numpy(normalize_1d(x_batch)).float()
            if CUDA:
                y1, x_batch, y_batch = y1.cuda(), x_batch.cuda(), y_batch.cuda()

            out = net(y1, x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
            loop.set_postfix(loss=loss.item())
    return vl.item(), pred_val, act_val


def test_predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        loop = tqdm(data_loader, desc=Colors.YELLOW + "Validate batches" + Colors.RESET,
                    total=len(data_loader), leave=True, ncols=100, ascii=False, colour='RED', unit='batch')
        for i, (x_batch, y_batch) in enumerate(loop):
            size_0 = x_batch.size(0)
            size_1 = x_batch.size(1)
            size_2 = x_batch.size(2)
            size_3 = x_batch.size(3)
            x_batch = x_batch.numpy()
            _, _, y1 = compute_stft(x_batch)  # y1[32,1,28,257,33]
            y1 = y1[:, :, :, 1:, :]
            y1 = y1.reshape(size_0*size_2, 1, 128, 8)
            y1 = normalize_2d(y1)
            y1 = torch.abs(torch.tensor(y1)).float()
            # y1 = torch.tensor(y1)
            y1 = y1.transpose(2, 3)
            y1 = y1.reshape(size_0, size_1, size_2, size_3)
            # y2 = x_batch.reshape(x_batch.size(0)*x_batch.size(2),10,256)
            x_batch = torch.from_numpy(normalize_1d(x_batch)).float()
            if CUDA:
                y1, x_batch, y_batch = y1.cuda(), x_batch.cuda(), y_batch.cuda()

            out = net(y1, x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
            loop.set_postfix(loss=loss.item())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):
       #   , data_test, label_test):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_trial' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)


    model = get_model(args)
    para = get_trainable_parameter_num(model)
    print('Model {} size:{}'.format(args.model, para))

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(
            args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )


        # scheduler.step()
        # for param_group in optimizer.param_groups:
        #     current_lr = param_group['lr']
        #     print(f'Epoch {epoch}/{args.max_epoch}, Current learning rate: {current_lr}')

        acc_train, f1_train, _ = get_metrics(
            y_pred=pred_train, y_true=act_train)
        print(Colors.GREEN + 'epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print(Colors.YELLOW + 'epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))


        if acc_val > trlog['max_acc']:
            trlog['max_acc'] = acc_val
            save_model('max-acc')

            if args.save_model:
                # save model here for reproduce
                model_name_reproduce = 'sub' + \
                    str(subject) + '_fold' + str(fold) + '.pth'
                data_type = 'model_{}_{}'.format(
                    args.dataset, args.data_format)
                save_path = osp.join(args.save_path, data_type)
                ensure_path(save_path)
                model_name_reproduce = osp.join(
                    save_path, model_name_reproduce)
                torch.save(model.state_dict(), model_name_reproduce)

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print(Colors.RESET + 'ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                               subject, fold))
    save_name_ = 'trlog' + save_name
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return trlog['max_acc']


def test(args, data, label, reproduce, subject, fold):
    seed_all(args.random_seed)
    set_up(args)

    test_loader = get_dataloader(data, label, args.batch_size, False)

    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + \
            str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}'.format(args.dataset, args.data_format)
        save_path = osp.join(args.save_path, data_type)
        ensure_path(save_path)
        model_name_reproduce = osp.join(save_path, model_name_reproduce)
        model.load_state_dict(torch.load(model_name_reproduce))
    else:
        model.load_state_dict(torch.load(args.load_path))
    loss, pred, act = test_predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print(Colors.RED + \
          '>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act, f1


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
