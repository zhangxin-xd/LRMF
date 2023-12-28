from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
import models
import numpy as np
import pickle
from scipy.spatial import distance
import pdb
import cv2
import math
from scipy.spatial import distance
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default = '/home/st2/Model_Compression',help='Path to dataset')
parser.add_argument('--dataset', type=str, default = 'cifar10',choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet20', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate_dist', type=float, default=0.2, help='the reducing ratio of pruning based on Distance')
parser.add_argument('--layer_begin', type=int, default=0, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=54, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=3, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=5, help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')
parser.add_argument('--limit', type=int, default=1, help='num of batch for pruning')
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Distance Pruning Rate: {}".format(args.rate_dist), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    if args.use_pretrain:
        if os.path.isfile(args.pretrain_path):
            print_log("=> loading pretrain model '{}'".format(args.pretrain_path), log)
        else:
            dir = '/data/cifar10_pretrain/'
            whole_path = dir + 'cifar10_' + args.arch + '_base'
            args.pretrain_path = whole_path + '/checkpoint.pth.tar'
            print_log("Pretrain path: {}".format(args.pretrain_path), log)
        pretrain = torch.load(args.pretrain_path)
        if args.use_state_dict:
            net.load_state_dict(pretrain['state_dict'])
        else:
            net = pretrain['state_dict']

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']

            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    m = Mask(net)
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
    print("total remaining ratio is %f" % (1-args.rate_dist))

    val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)

    print(" accu before is: %.3f %%" % val_acc_1)
    global feature_ini_out_hook
    feature_ini_out_hook = []
    m.model = net
    for part in net.module.modules():
        print(part)
        if (isinstance(part,torch.nn.Conv2d)):
                handel = part.register_forward_hook(hook_ini)
                y = inference(train_loader,net)
                handel.remove()
    m.init_mask(args.rate_dist, feature_ini_out_hook, args.limit)
    m.if_zero()
    m.do_similar_mask()
    net = m.model
    m.if_zero()
    if args.use_cuda:
        net = net.cuda()
    val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)
    print(" accu after is: %s %%" % val_acc_2)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()


    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log, m)

        # evaluate on validation set
        val_acc_1, val_los_1 = validate(test_loader, net, criterion, log)
        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            
            global feature_out_hook 
            feature_out_hook = []
            m.model = net

            for part in net.module.modules():
                print(part)
                if (isinstance(part,torch.nn.Conv2d)):
                    handel = part.register_forward_hook(hook)
                    y = inference(train_loader,net)
                    handel.remove()
       
            m.if_zero()
            m.init_mask(args.rate_dist, feature_out_hook, args.limit)
            m.do_similar_mask()
            m.if_zero()
            net = m.model
            if args.use_cuda:
                net = net.cuda()

        val_acc_2, val_los_2 = validate(test_loader, net, criterion, log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

    log.close()

def hook_ini(module, inputdata,output):
    print(output.data.size())
    feature_ini_out_hook.append(output.data.cpu().numpy())   
def hook(module, inputdata,output):
    print(output.data.size())   
    feature_out_hook.append(output.data.cpu().numpy())
    
# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, m):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        m.do_grad_mask()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item (), input.size(0))
        top1.update(prec1.item (), input.size(0))
        top5.update(prec5.item (), input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                   error1=100 - top1.avg),
              log)

    return top1.avg, losses.avg

def inference(train_loader, model):

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            if (i >= args.limit):
                break
            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda()
            output = model(input)

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}


    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
         
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, distance_rate, length, feature_out_hook_conv):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:  
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            #weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            num_batch_prune = len(feature_out_hook_conv)
            bs = feature_out_hook_conv[0].shape[0]
            channel = feature_out_hook_conv[0].shape[1]
            w = feature_out_hook_conv[0].shape[2]
            a = math.ceil(w/4)
            distance_final = torch.zeros(channel)
            for i in range(num_batch_prune):
                distance_all = []
                for j in range(bs):
                    feature_out_hook_conv1 = feature_out_hook_conv[i][j]
                    fd = []
                    for k in range(channel):
                        feature_channel = np.resize(feature_out_hook_conv1[k, :, :],(w, w))
                        feature_dct = cv2.dct(feature_channel)                   
                        vis = feature_dct[:a,:a]             
                        fd.append(vis)
                    fd_np = np.array(fd).reshape(channel,-1)
                    distance_feature = distance.cdist(fd_np,fd_np, 'euclidean')
                    distance_sum = np.sum(np.abs(distance_feature), axis=0 )
                    distance_all.append(distance_sum)
                distance_all = np.array(distance_all)
                distance_all = torch.from_numpy(distance_all)
                distance_all = distance_all.view(bs,-1).float()
                distance_all = distance_all.sum(0)
                distance_final = distance_final.float() + distance_all
            distance_final = distance_final/num_batch_prune
            distance_final = distance_final.numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = distance_final.argsort()[similar_pruned_num:]
            filter_small_index = distance_final.argsort()[:similar_pruned_num]

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_small_index)):
                codebook[
                filter_small_index[x] * kernel_length: (filter_small_index[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, rate_dist_per_layer):
        for index, item in enumerate(self.model.parameters()):
            self.distance_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.distance_rate[key] = rate_dist_per_layer
        # different setting for  different architecture
        if args.arch == 'resnet20':
            last_index = 57
        elif args.arch == 'resnet32':
            last_index = 93
        elif args.arch == 'resnet56':
            last_index = 165
        elif args.arch == 'resnet110':
            last_index = 327
        # to jump the last fc layer
        self.mask_index = [x for x in range(0, last_index, 3)]

    #        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, rate_dist_per_layer,feature_out_hook,num):
        self.init_rate(rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                index_feature_begin = int((index/3)*num)
                index_feature_end = int(index_feature_begin+num)
                feature_out_hook_conv = feature_out_hook[index_feature_begin:index_feature_end]

                self.similar_matrix[index] = self.get_filter_similar(item.data,
                                                                     self.distance_rate[index],
                                                                     self.model_length[index],feature_out_hook_conv)                
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")


    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
