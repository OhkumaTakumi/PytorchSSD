from __future__ import print_function

import argparse
import pickle
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer

from matplotlib import  pyplot as plt
from Videoframe import Videoframes
from pseudoframe import Pseudoframes

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')

parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='VOC or COCO dataset or youtube')
parser.add_argument(
    '--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=16,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-v', '--version', default='SSD_vgg',
                    help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.')
parser.add_argument('-max', '--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('-we', '--warm_epoch', default=3,
                    type=int, help='max epoch for retraining')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--date', default='1213')
parser.add_argument('--save_frequency', default=10)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency', default=10)
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--classes', default='all',
                    help='all or youtube_bb or youtube_bb_sub')
parser.add_argument('--box_num', default=100000000,
                    help='max number of bounding boxes of each classes or "full" or default')
parser.add_argument('--test_only', default=False,
                    type=bool, help='Only test not train')
parser.add_argument('--test_eval', default=True,
                    type=bool, help='Evaluating on test data')
parser.add_argument('--pseudo', default=False,
                    type=bool, help='Use pseudo labels from video')
args = parser.parse_args()

path_result = "/home/takumi/research/PytorchSSD/result"
if not args.test_only:
    with open(os.path.join(path_result, "result_{0}boxes.txt".format(args.box_num)), mode='w') as result_file:
        result_file.write("result of the situation that the number of boxes of each classes are {0}\n\n".format(args.box_num))

save_folder = os.path.join(args.save_folder, args.version + '_' + args.size, args.date)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
test_save_dir = os.path.join(save_folder, 'ss_predict')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

log_file_path = save_folder + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    train_sets = [('2017', 'train')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net

    cfg = COCO_mobile_300
elif args.version == 'SSD_vgg':
    from models.SSD_vgg import build_net
elif args.version == 'FSSD_vgg':
    from models.FSSD_vgg import build_net
elif args.version == 'FRFBSSD_vgg':
    from models.FRFBSSD_vgg import build_net
else:
    print('Unkown version!')
rgb_std = (1, 1, 1)
img_dim = (300, 512)[args.size == '512']
if 'vgg' in args.version:
    rgb_means = (104, 117, 123)
elif 'mobile' in args.version:
    rgb_means = (103.94, 116.78, 123.68)

p = (0.6, 0.2)[args.version == 'RFB_mobile']
num_classes = (21, 81)[args.dataset == 'COCO']
if args.classes == "youtube_bb" and (args.dataset == 'COCO' or args.dataset == 'youtube'):
    num_classes = 23
if args.classes == "youtube_bb_sub" and (args.dataset == 'COCO' or args.dataset == 'youtube'):
    num_classes = 6


batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9
if args.visdom:
    import visdom

    viz = visdom.Visdom()

net = build_net(img_dim, num_classes)
print(net)
if not args.resume_net:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)


    def xavier(param):
        init.xavier_uniform(param)


    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    if args.version == 'FSSD_vgg' or args.version == 'FRFBSSD_vgg':
        net.ft_module.apply(weights_init)
        net.pyramid_ext.apply(weights_init)
    if 'RFB' in args.version:
        net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
    # load resume network
    resume_net_path = os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' + \
                                   str(args.resume_epoch) + '.pth')

    resume_net_path = "/home/takumi/important_result/RFB_SSD/1000_Final_RFB_vgg_COCO.pth"

    print('Loading resume network', resume_net_path)
    state_dict = torch.load(resume_net_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

net_size_copy = net.size
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))



if args.cuda:
    net.cuda()
    cudnn.benchmark = True

detector = Detect(num_classes, 0, cfg)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)
# dataset

if args.dataset == 'VOC':
    print('Loading Dataset...')
    testset = VOCDetection(
        VOCroot, [('2007', 'test')], None, AnnotationTransform())
    train_dataset = VOCDetection(VOCroot, train_sets, preproc(
        img_dim, rgb_means, rgb_std, p), AnnotationTransform())
elif args.dataset == 'COCO':
    print('Loading Dataset...')
    if args.classes == 'youtube_bb':
        testset = COCODetection(
            COCOroot, [('2017', 'val')], None, classes='youtube_bb')
        train_dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, rgb_std, p), classes='youtube_bb',  box_num=args.box_num)
    elif args.classes == 'youtube_bb_sub':
        testset = COCODetection(
            COCOroot, [('2017', 'val')], None, classes='youtube_bb_sub')
        train_dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, rgb_std, p), classes='youtube_bb_sub',  box_num=args.box_num)
    else:
        testset = COCODetection(
            COCOroot, [('2017', 'val')], None)
        train_dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, rgb_std, p))
elif args.dataset != 'youtube':
    print('Only VOC ,COCO and youtube are supported now!')
    exit()


def train():

    epoch = 0

    if args.test_only:
        net.eval()
        top_k = (300, 200)[args.dataset == 'COCO']
        if args.dataset == 'VOC':
            APs, mAP = test_net(test_save_dir, net, detector, args.cuda, testset,
                                BaseTransform(net.module.size, rgb_means, rgb_std, (2, 0, 1)),
                                top_k, thresh=0.01, epoch=epoch)
            APs = [str(num) for num in APs]
            mAP = str(mAP)
            log_file.write(str(iteration) + ' APs:\n' + '\n'.join(APs))
            log_file.write('mAP:\n' + mAP + '\n')
        elif args.dataset == "COCO" or args.dataset == "youtube":
            test_net(test_save_dir, net, detector, args.cuda, testset,
                     BaseTransform(net_size_copy, rgb_means, rgb_std, (2, 0, 1)),
                     top_k, thresh=0.01, epoch=epoch)
        return

    net.train()
    # loss counters

    if args.resume_net:
        epoch = 0 + args.resume_epoch
    epoch_size = len(train_dataset) // args.batch_size

    #epoch_based
    #max_iter = args.max_epoch * epoch_size

    #iteration_based
    max_iter = 100000


    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    #stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues_COCO = (60000, 75000, 90000)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
    print('Training', args.version, 'on', train_dataset.name)
    step_index = 0

    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),


            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0



    log_file = open(log_file_path, 'w')
    batch_iterator = None
    mean_loss_c = 0
    mean_loss_l = 0

    closs_list=[]
    lloss_list=[]
    loss_list=[]
    itra_list=[]
    # setting of graph
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    plt.title('Simple Curve Graph')  ## グラフタイトル（必須ではない）
    plt.xlabel('iteration')  ## x軸ラベル（必須ではない）
    plt.ylabel('loss')  ## y軸ラベル（必須ではない）
    plt.ylim(0, 25)  ## y軸範囲固定（必須ではない）
    plt.grid()

    #epoch_size = len(pseudo_dataset) // args.batch_size

    for iteration in range(start_iter, max_iter + 10):
        if iteration > 0 and iteration % 1000 == 0:
            with open(os.path.join(path_result, "{0}boxes_result.txt".format(args.box_num)),
                      mode='a') as result_file:
                result_file.write("iteration : {0}\n".format(iteration))
                result_file.write("location loss   : {0}\n".format(closs_list[-1]))
                result_file.write("confidence loss : {0}\n".format(lloss_list[-1]))
                result_file.write("all loss        : {0}\n".format(loss_list[-1]))
        if (iteration % epoch_size == 0):

            batch_iterator = iter(data.DataLoader(train_dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            #pseudo_iterator = iter(data.DataLoader(pseudo_dataset, batch_size,
            #                                      shuffle=True, num_workers=args.num_workers,
            #                                      collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            if epoch % args.save_frequency == 0 and epoch > 0:
                torch.save(net.state_dict(), os.path.join(save_folder, args.version + '_' + args.dataset +
                                                          '_{0}_boxes'.format(args.box_num) + '_epoches_' + repr(epoch) + '.pth'))
            if epoch % args.test_frequency == 0 and epoch > 0:
                net.eval()
                top_k = (300, 200)[args.dataset == 'COCO']
                if args.dataset == 'VOC':
                    APs, mAP = test_net(test_save_dir, net, detector, args.cuda, testset,
                                        BaseTransform(net.module.size, rgb_means, rgb_std, (2, 0, 1)),
                                        top_k, thresh=0.01, epoch=epoch)
                    APs = [str(num) for num in APs]
                    mAP = str(mAP)
                    log_file.write(str(iteration) + ' APs:\n' + '\n'.join(APs))
                    log_file.write('mAP:\n' + mAP + '\n')
                else:
                    test_net(test_save_dir, net, detector, args.cuda, testset,
                             BaseTransform(net_size_copy, rgb_means, rgb_std, (2, 0, 1)),
                             top_k, thresh=0.01, epoch=epoch)

                net.train()
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index = stepvalues.index(iteration) + 1
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([mean_loss_l, mean_loss_c,
                                    mean_loss_l + mean_loss_c]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)


        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        # arm branch loss
        loss_l, loss_c = criterion(out, priors, targets)
        # odm branch loss

        mean_loss_c += loss_c
        mean_loss_l += loss_l

        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        if iteration % 100 == 0 and iteration != 0:
            closs_list.append(mean_loss_c.item()/10)
            lloss_list.append(mean_loss_l.item()/10)
            loss_list.append(mean_loss_c.item()/10 + mean_loss_l.item()/10)
            itra_list.append((iteration))
            # 描画領域
            plt.plot(itra_list, closs_list, color='blue')
            plt.plot(itra_list, lloss_list, color='green')
            plt.plot(itra_list, loss_list, color='red')
            plt.draw()
            plt.pause(0.001)
            loss_all=[closs_list, lloss_list, loss_list, itra_list]

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f||' % (
                      mean_loss_l / 10, mean_loss_c / 10) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            log_file.write(
                'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                + '|| Totel iter ' +
                repr(iteration) + ' || L: %.4f C: %.4f||' % (
                    mean_loss_l / 10, mean_loss_c / 10) +
                'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr) + '\n')

            mean_loss_c = 0
            mean_loss_l = 0

            #save loss
            loss_all = [closs_list, lloss_list, loss_list]
            with open("/home/takumi/research/PytorchSSD/result/{0}_box.binaryfile".format(args.box_num), "wb") as f:
                pickle.dump(loss_all, f)

            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())

    log_file.close()
    torch.save(net.state_dict(), os.path.join(save_folder,
                                              '{0}'.format(args.box_num) + 'Final_' + args.version + '_' + args.dataset + '.pth'))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warm_epoch:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * args.warm_epoch)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005, epoch=0):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]


    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')
    path_file = os.path.join(save_folder, 'img_path_coco2017_val.pkl')
    if epoch == args.max_epoch or args.test_only:
        det_file = os.path.join(save_folder, 'detections{0}.pkl'.format(args.box_num))
    if args.dataset == "youtube":
        det_file = path1 + path_result + '.pkl'

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    img_path = []

    for i in range(num_images):
        if args.dataset != "youtube":
            img_path.append([i, testset.ids[i]])
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        boxes, scores = detector.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            if args.dataset == 'VOC':
                cpu = False
            else:
                cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]

            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    if args.dataset == "youtube":
        return

    #save img path
    with open(path_file, 'wb') as f:
        pickle.dump(img_path, f, pickle.HIGHEST_PROTOCOL)

    if not args.test_eval:
        return

    print('Evaluating detections')
    if args.dataset == 'VOC':
        APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
        return APs, mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)

    with open('/home/takumi/research/PytorchSSD/weights/RFB_vgg_512/1213/ss_predict/detection_results.pkl', 'rb') as f:
        data = pickle.load(f)

    if args.test_only:
        return

    with open(os.path.join(path_result, "{0}boxes_result.txt".format(args.box_num)), mode='a') as result_file:

        result_file.write(
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {0}\n".format(data.stats[0]))
        result_file.write(
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {0}\n".format(data.stats[1]))
        result_file.write(
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {0}\n".format(data.stats[2]))
        result_file.write(
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {0}\n".format(data.stats[3]))
        result_file.write(
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {0}\n".format(data.stats[4]))
        result_file.write(
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {0}\n".format(data.stats[5]))
        result_file.write(
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1   ] = {0}\n".format(data.stats[6]))
        result_file.write(
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=10  ] = {0}\n".format(data.stats[7]))
        result_file.write(
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {0}\n".format(data.stats[8]))
        result_file.write(
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {0}\n".format(data.stats[9]))
        result_file.write(
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {0}\n".format(data.stats[10]))
        result_file.write(
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {0}\n".format(data.stats[11]))
        result_file.write("\n\n")


if __name__ == '__main__':
    if args.dataset != "youtube":
        if args.pseudo:
            pseudo_dataset = Pseudoframes(preproc=preproc(img_dim, rgb_means, rgb_std, p))
        train()
    else:
        for i in range(200, 50000):
            for class_num in range(1,24):
                if class_num == 22:
                    continue
                path1 = "/home/takumi/data/YouTube-BB"
                path_classfolder = "/home/takumi/data/YouTube-BB/videos/{0}".format(class_num)

                video_list = os.listdir(path_classfolder)
                if len(video_list) > i:
                    video = video_list[i]

                    path_video = "/videos/{0}/".format(class_num) + video
                    path_result = "/detection_result/{0}/".format(class_num) + video[:-4]

                    print(i, class_num)

                    testset = Videoframes(path1 + path_video)
                    train()




