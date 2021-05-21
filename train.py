#!/usr/bin/python3

import argparse
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import numpy as np
import shutil
import cv2
import random
from torch.utils.data import DataLoader
from models import trgb_segnet as models
from models import build_net
import thermal_loader
from models import conf_segnet
import utils
import vis_utils
from evaluate import validate_model


def createDataloader(test_stamps=None):
    dataloader_train = thermal_loader.ThermalDataLoader(opt.dataroot, split='train', test_stamps=test_stamps)

    train_loader = torch.utils.data.DataLoader(dataloader_train,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers,
                                               pin_memory=True,
                                               drop_last=True)
    return train_loader


def createValloader(data_dirnames):

    dataloader_val = thermal_loader.ThermalTestDataLoader(*utils.getPaths(data_dirnames))

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=opt.num_workers,
                                               pin_memory=True,
                                               drop_last=False)
    return val_loader


def createMFNetValloader(root_dir, split):

    dataloader_val = thermal_loader.MFDataset(root_dir, split=split)

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=opt.num_workers,
                                               pin_memory=True,
                                               drop_last=False)

    return val_loader




def rectDropTensor(tensor, params):
    params = params.int()
    for i in range(tensor.size(0)):
        tensor[i, :, params[i,0]:(params[i,0]+params[i,2]), params[i,1]:(params[i,1]+params[i,3])] = 0
    return tensor

def getTestStamps(ir_files, rgb_files, label_files):
    stamps = []
    for filename in label_files:
        split_path = filename.split('_')
        digits = []
        for s in split_path:
            if s.isdigit():
                digits.append(int(s))

        stamps.append((digits[0], digits[1]))
    return stamps


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')
        wandb.save(filename + '_best.pth.tar')


class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


wandb.init(project="hotnet", entity='team-awesome')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--dataroot', type=str, default='/mnt/hpc.shared/ir_rgb_data/', help='root directory of the dataset')
parser.add_argument('--testroot_day', type=str, default='/mnt/hpc.shared/label_data/test_set_day/converted/', help='root directory of the day dataset')
parser.add_argument('--testroot_night', type=str, default='/mnt/hpc.shared/label_data/test_set_night/converted/', help='root directory of the night dataset')
parser.add_argument('--discarch', type=str, default='cyclegan', help='name of the critic architecture')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--lr_disc', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--conf_weight', type=float, default=0.1, help='weight for confusion')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
parser.add_argument('--iter_seg_phase', type=int, default=50, help='iterations for training the semantic network')
parser.add_argument('--iter_critic_phase', type=int, default=500, help='iterations for training the critic')
parser.add_argument('--iter_initial_critic_phase', type=int, default=1000, help='iterations for training the critic')
parser.add_argument('--moddrop', action='store_true', help='enables modality rectangle dropout')
parser.add_argument('--irscale', action='store_true', help='enables ir scale augmentation')
parser.add_argument('--no_conf', action='store_true', help='disables confusion training /activates simple cross-entropy training')
parser.add_argument('--vis', action='store_true', help='enables visualization')
parser.add_argument('--gpus', nargs='+', type=int, help='indeces of gpus')
parser.add_argument('--num_critics', type=int, default=6, help='number of critics')
parser.add_argument('--half_lr_every_epoch', type=int, default=30, help='the learning rate is halfed every N epochs')
parser.add_argument('--feedback_seg', action='store_true', help='add seg to every critic')
parser.add_argument('--checkpointname', type=str, default='checkpoint', help='name of file for saving the model')
parser.add_argument('--modalities', default='ir_rgb', type=str, help='names of used modalities')
parser.add_argument('--pretraining', action='store_true', help='use pretrained segmentation network')
parser.add_argument('--night_supervision_model', type=str, default="", help='use pretrained ir-only network as night supervision')
parser.add_argument('--night_supervision_model_modalities', type=str, default="", help='specifies the used modalities for the night supervision model')
parser.add_argument('--resume', type=str, default="", help='initialize with given model parameters')
parser.add_argument('--train_input_adapter', action='store_true', help='attach input adapter and train it')
parser.add_argument('--cert_branch', action='store_true', help='adds training of certainty for day part')
parser.add_argument('--weight_ir_sup', action='store_true', help='weights confusion gradients according to the ir-uncertainty')
parser.add_argument('--late_fusion', action='store_true', help='late-fusion instead of early-fusion for multimodal training, only considered if multiple modalities are trained')
parser.add_argument('--arch', default='custom', type=str, help='custom')
parser.add_argument('--critic_weights', nargs='+', type=float, default=[1., 1., 1., 1., 1., 1., 1.], help='loss weight for individual critics')
parser.add_argument('--adv_loss', default='MSE', type=str, help='loss function for adversarial term')
parser.add_argument('--multidir', action='store_true', help='dual gan')
parser.add_argument('--trainsetname', type=str, default="FR", help='name of the training dataset')
parser.add_argument('--eval', type=str, default="", help='if name of a dataset is passed, a evaluation on the respective dataset gets started')
parser.add_argument('--im_save_dir', type=str, default="", help='if eval mode: images are saved in this dir')

opt = parser.parse_args()

best_iou = 0.0

if opt.eval is not "":
    print('##############EVALUATING MODE##############')

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
gpus = opt.gpus
torch.cuda.set_device(gpus[0])

# Create model
conf_segnet_model = conf_segnet.conv_segnet(pretrained=opt.pretraining,
                                            disc_arch=opt.discarch,
                                            num_critics=opt.num_critics,
                                            feedback_seg=opt.feedback_seg,
                                            no_conf=opt.no_conf,
                                            modalities=opt.modalities,
                                            input_adapter=opt.train_input_adapter,
                                            cert_branch=opt.cert_branch,
                                            arch=opt.arch,
                                            late_fusion=opt.late_fusion)

night_supervision_active = False

if opt.night_supervision_model is not "":
    night_sup_num_input_channels = 0
    if 'rgb' in opt.night_supervision_model_modalities:
        night_sup_num_input_channels += 3
        print('Using RGB for night supervision model')

    if 'ir' in opt.night_supervision_model_modalities:
        night_sup_num_input_channels += 1
        print('Using IR for night supervision model')

    if opt.arch == "custom":
        ir_only_segnet = models.ResNeXt(**{"structure": [3, 4, 6, 3], "input_channels": 1, "cert_branch": opt.cert_branch})
    elif opt.arch == "pspnet":
        ir_only_segnet = build_net.build_network(None, 'resnet50', in_channels=night_sup_num_input_channels,
                                                   late_fusion=False if night_sup_num_input_channels < 4 else True)

    ir_only_segnet.apply(utils.weights_init_normal)
    night_supervision_active = True
    ir_only_segnet = nn.DataParallel(ir_only_segnet.cuda(), gpus)
    utils.initModelRenamed(ir_only_segnet, opt.night_supervision_model, "module.trgb_segnet.", "module.")

if opt.cuda:
    conf_segnet_model.cuda()

conf_segnet_model = nn.DataParallel(conf_segnet_model.cuda(), gpus)

# Lossess
if opt.adv_loss == "BCE":
    criterion_conf = torch.nn.BCEWithLogitsLoss()
elif opt.adv_loss == "MSE":
    criterion_conf = torch.nn.MSELoss()
else:
    print("Loss not known : %s " % (opt.adv_loss))

criterion_semseg = torch.nn.CrossEntropyLoss()
criterion_semseg_weighted = torch.nn.CrossEntropyLoss(reduction='none')

if not opt.no_conf:
   critics_params = []
   for p in conf_segnet_model.module.critics:
     critics_params.append({'params': p.parameters()})
   if opt.feedback_seg:
     for p in conf_segnet_model.module.downscale_nets:
       critics_params.append({'params': p.parameters()})


# change number of epochs if we train both segmentation and critics
if not opt.no_conf:
    opt.n_epochs *= 2
    opt.half_lr_every_epoch *= 2


optimizer = torch.optim.RMSprop(conf_segnet_model.parameters(), lr=opt.lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.half_lr_every_epoch, gamma=0.5)

if opt.resume is not "":
    checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage)
    conf_segnet_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    opt.epoch = checkpoint['epoch']
    best_iou = checkpoint['best_iou']

print('Create validation loader daytime')
val_loader_night = createValloader([opt.testroot_night])

print('Create validation loader nighttime')
val_loader_day = createValloader([opt.testroot_day])

print('Create validation loader both')
val_loader_combined = createValloader([opt.testroot_night, opt.testroot_day])

test_stamps = getTestStamps(*utils.getPaths([opt.testroot_night, opt.testroot_day]))

print('Create training loader')
train_loader = createDataloader(test_stamps)


# Loss plot
total_loss_avgmeter_phase1 = AverageMeter()
total_loss_avgmeter_phase2 = AverageMeter()
critic_loss_avgmeter = AverageMeter()
seg_loss_avgmeter = AverageMeter()
conf_loss_avgmeter = AverageMeter()


if opt.eval is not "":
    print('Starting evaluation on: %s....' % (opt.eval))
    night_split = True if "night" in opt.eval else False

    if "FR" in opt.eval:
        iou_fr = validate_model(conf_segnet_model.module.trgb_segnet, val_loader_night if night_split else val_loader_day,
                                                      opt.modalities,
                                                      mode="night" if night_split else "day", vis=opt.vis, save_dir=opt.im_save_dir)

    else:
        print('Eval dataset %s not known... exiting' % (opt.eval))
        exit()

    print('Eval successful!')
    exit()

###### Training ######

state = "train_critic"
counter = opt.iter_initial_critic_phase

if opt.no_conf:
    state = "train_seg"
conf_segnet_model.module.setPhase(state)

drop_mod_activate = opt.moddrop
ir_scale_aug_activate = opt.irscale

if opt.vis:
    color_coder = vis_utils.ColorCode(13)

drop_mod = drop_mod_activate if state == "train_seg" else False
ir_scale_aug = ir_scale_aug_activate if state == "train_seg" else False

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_loader):

        if i > 500:
            break

        print('Epoch {}/{} - Iteration {}/{}'.format(epoch, opt.n_epochs, i, len(train_loader)))

        rgb_day = batch['rgb_day'].cuda()
        ir_day = batch['ir_day'].cuda()
        rgb_night = batch['rgb_night'].cuda()
        ir_night = batch['ir_night'].cuda()
        label_day = batch['label_day'].cuda().long()
        if opt.trainsetname == 'FR':
            mod_drop_params = batch['mod_drop_params']

        if drop_mod:
            if bool(random.getrandbits(1)):
                # First select modality for rect drop
                drop_rgb = bool(random.getrandbits(1))
                if drop_rgb:
                    drop_rgb = rectDropTensor(rgb_day, mod_drop_params)
                else:
                    drop_rgb = rectDropTensor(ir_day, mod_drop_params)

        if ir_scale_aug:
            if bool(random.getrandbits(1)):
                scale = random.uniform(0.1, 1)
                ir_day = scale * ir_day

        optimizer.zero_grad()

        # Arrange input depending on selected modalities
        if 'rgb' in opt.modalities and 'ir' in opt.modalities:
            in_day = [rgb_day, ir_day]
            in_night = [rgb_night, ir_night]
        elif 'rgb' in opt.modalities:
            in_day = [rgb_day]
            in_night = [rgb_night]
        elif 'ir' in opt.modalities:
            in_day = [ir_day]
            in_night = [ir_night]
        else:
            print('No known modality selected....')
            exit()

        out_model = conf_segnet_model(in_day, in_night)

        if night_supervision_active:
            with torch.no_grad():
              night_sup_input = [ir_night] if night_sup_num_input_channels < 4 else [rgb_night, ir_night]
              out_night_ir_only, _, ir_only_cert = ir_only_segnet(*night_sup_input)
              out_night_ir_only = F.softmax(out_night_ir_only)
            torch.cuda.synchronize()

        if not opt.no_conf:
            total_critics_a = torch.zeros(1).cuda()
            total_critics_b = torch.zeros(1).cuda()
            for c_a in out_model['critics_a']:
                total_critics_a += torch.sum(criterion_conf(c_a, torch.full_like(c_a, 1)))

            for c_b in out_model['critics_b']:
                total_critics_b += torch.sum(criterion_conf(c_b, torch.full_like(c_b, 0)))

            total_critics = total_critics_a + total_critics_b

        if state == "train_seg":
            # This phase trains the semantic output while aiming for uncertain outputs for the critic

            # Semantic loss
            seg_loss = criterion_semseg(out_model['pred_label_a'], label_day)
            print('Day Seg loss: %f' % (seg_loss))
            if night_supervision_active:
                if not opt.weight_ir_sup:
                    seg_loss_ir_only = criterion_semseg(out_model['pred_label_b'], torch.argmax(out_night_ir_only, 1).squeeze().long())
                    if opt.vis:
                        # vis_utils.visDepth(out_night_ir_only[0:1, ...].max(1)[1].float(), 'night_ir_label')
                        night_sup_label_colored = color_coder.color_code_labels(out_night_ir_only[0:1, ...], True)
                        cv2.imshow('sup_night_label', night_sup_label_colored)

                elif opt.weight_ir_sup and opt.cert_branch:
                    seg_loss_ir_only = criterion_semseg_weighted(out_model['pred_label_b'], torch.argmax(out_night_ir_only, 1).squeeze().long())
                    seg_loss_ir_only = torch.mean((torch.ones_like(ir_only_cert) - ir_only_cert) * seg_loss_ir_only)
                    if opt.vis:
                        vis_utils.visDepth(ir_only_cert[0:1, ...], 'weighting_ir')
                        vis_utils.visDepth(out_night_ir_only[0:1, ...].max(1)[1].float(), 'night_ir_label')
                else:
                    seg_loss_ir_only = criterion_semseg_weighted(out_model['pred_label_b'],
                                                                 torch.argmax(out_night_ir_only, 1).squeeze(1).long())
                    cert = F.softmax(out_night_ir_only)
                    cert = cert.max(1)[0]

                    if opt.vis:
                        vis_utils.visDepth(cert[0:1, ...],  'weighting_ir')
                        vis_utils.visDepth(out_night_ir_only[0:1,...].max(1)[1].float(), 'night_ir_label')
                    seg_loss_ir_only = torch.mean(cert * seg_loss_ir_only)

                seg_loss += seg_loss_ir_only
                print('Night Seg loss: %f' % seg_loss_ir_only)

            if opt.cert_branch and not night_supervision_active:
                one_hot_label = utils.make_one_hot(label_day.unsqueeze(1), out_model['pred_label_a'].size(1))
                cert = torch.sum((one_hot_label.float() * F.softmax(out_model['pred_label_a'])), 1)
                cert = torch.ones_like(cert) - cert
                if opt.vis:
                    vis_utils.visDepth(cert[0:1, ...], 'cert_gt')

                cert_loss = torch.mean((out_model['cert_a'] - cert)**2) * 10
                print('Cert_loss : %f , Seg loss: %f' % (cert_loss, seg_loss))
                seg_loss += cert_loss

            # Visualize training images
            if opt.vis:
                vis_utils.visImage3Chan(rgb_day[0:1,...], 'rgb_day')
                # vis_utils.visDepth(label_day[0:1, ...].float(), 'day_label')
                day_label_colored = color_coder.color_code_labels(label_day[0:1, ...].unsqueeze(0), False)
                cv2.imshow('sup_day_label', day_label_colored)
                if opt.train_input_adapter:
                    vis_utils.visImage3Chan(F.sigmoid(out_model['input_a'][0:1,...]), 'adapted_day')
                    vis_utils.visImage3Chan(F.sigmoid(out_model['input_b'][0:1,...]), 'adapted_night')

                if opt.no_conf:
                    vis_utils.visDepth(out_model['pred_label_a'][0:1, ...].max(1)[1].unsqueeze(0).float(),
                                         'label_day')
                    vis_utils.visDepth(out_model['pred_label_b'][0:1, ...].max(1)[1].unsqueeze(0).float(),
                                       'label_night')
                else:
                    critic_vals_a = [torch.mean(p).gt(0.5).item() for p in out_model['critics_a']]
                    critic_vals_b = [torch.mean(p).gt(0.5).item() for p in out_model['critics_b']]
                    vis_utils.visSegDisc(out_model['pred_label_a'][0:1,...].max(1)[1].unsqueeze(0).float(), 'label_day', critic_vals_a)
                    vis_utils.visSegDisc(out_model['pred_label_b'][0:1,...].max(1)[1].unsqueeze(0).float(), 'label_night', critic_vals_b)
                    vis_utils.visImage3Chan(rgb_night[0:1, ...], 'rgb_night')

                if opt.cert_branch:
                    vis_utils.visDepth(out_model['cert_a'][0:1, ...], 'cert_a')

                if 'ir' in opt.modalities:
                    vis_utils.visDepth(ir_day[0:1, ...], 'ir_day')
                    if not opt.no_conf:
                        vis_utils.visDepth(ir_night[0:1, ...], 'ir_night')

                cv2.waitKey(10)

            # Critic loss
            if opt.no_conf:
                total_loss = seg_loss
            else:
                weights = opt.critic_weights
                conf_loss = torch.zeros(1).cuda()
                if opt.weight_ir_sup and night_supervision_active:
                    conf_weighting = (torch.full_like(cert, 1.0)-cert).unsqueeze(0)
                else:
                    conf_weighting = torch.ones_like(out_model['critics_a'][0])

                for m, c_a in enumerate(out_model['critics_a']):
                    conf_loss += torch.mean(F.interpolate(conf_weighting, size=(c_a.size(2), c_a.size(3)),
                                                          mode="bilinear") * criterion_conf(c_a, torch.full_like(c_a, 0) if opt.multidir else torch.full_like(c_a, 1))) * weights[m]

                for m, c_b in enumerate(out_model['critics_b']):
                    conf_loss += torch.mean(F.interpolate(conf_weighting, size=(c_b.size(2), c_b.size(3)),
                                                          mode="bilinear") * criterion_conf(c_b, torch.full_like(c_b, 1))) * weights[m]

                conf = conf_loss
                print('Conf loss: %f ' % (conf_loss))

                total_loss = (seg_loss + opt.conf_weight*conf)

            total_loss.backward()
            optimizer.step()

            total_loss_avgmeter_phase1.update(total_loss.item())
            seg_loss_avgmeter.update(seg_loss.item())

            if not opt.no_conf:
                conf_loss_avgmeter.update(conf.item())
            else:
                conf_loss = torch.zeros(1)

            print("Current loss: %f " % total_loss_avgmeter_phase1.avg)
            wandb.log({'epoch': epoch, 'total_loss_phase1': total_loss_avgmeter_phase1.avg, 'seg_loss': seg_loss_avgmeter.avg,
                       'conf_loss': (conf_loss).item()})

        elif state == "train_critic":
            # This phase trains the critic to distinguish between the states of the input
            total_loss = total_critics

            total_loss.backward()
            optimizer.step()

            total_loss_avgmeter_phase2.update(total_loss.item())
            critic_loss_avgmeter.update((total_loss).item())

            print("Current loss: %f " % total_loss_avgmeter_phase2.avg)
            wandb.log({'epoch': epoch, 'total_loss_phase2': total_loss_avgmeter_phase2.avg, 'critic_loss': critic_loss_avgmeter.avg})

        # Swich learning phase ################################
        if not opt.no_conf:
            counter = counter - 1

            if counter == 0:
                if state == "train_seg":
                    state = "train_critic"
                    counter = opt.iter_critic_phase
                    drop_mod = False
                    ir_scale_aug = False

                elif state == "train_critic":
                    state = "train_seg"
                    counter = opt.iter_seg_phase
                    drop_mod = drop_mod_activate
                    ir_scale_aug = ir_scale_aug_activate
                conf_segnet_model.module.setPhase(state)

    # Update learning rates
    lr_scheduler.step()
    eval_everyn = 2 if opt.trainsetname == 'FR' else 20

    if (epoch % eval_everyn) == 0:

        # Evaluate night images
        ious_night = validate_model(conf_segnet_model.module.trgb_segnet, val_loader_night, opt.modalities, mode="night")
        # Evaluate day images
        ious_day = validate_model(conf_segnet_model.module.trgb_segnet, val_loader_day, opt.modalities, mode="day")

        ious_combined = (ious_day + ious_night) / 2
        iou_combined_mean = np.mean(ious_combined)

        # log combined ious
        wandb.log({
            "combined_Test mean IoU": iou_combined_mean,
            "combined_Test IoU road,parking": ious_combined[0],
            "combined_Test IoU ground,sidewalk": ious_combined[1],
            "combined_Test IoU building,": ious_combined[2],
            'combined_Test IoU curb': ious_combined[3],
            'combined_Test IoU fence': ious_combined[4],
            'combined_Test IoU pole,traffic light,traffic sign': ious_combined[5],
            'combined_Test IoU vegetation': ious_combined[6],
            'combined_Test IoU terrain': ious_combined[7],
            'combined_Test IoU sky': ious_combined[8],
            'combined_Test IoU person,rider': ious_combined[9],
            'combined_Test IoU car,truck,bus,train': ious_combined[10],
            'combined_Test IoU motorcycle,bicycle': ious_combined[11],
        })

        is_best = False
        if iou_combined_mean > best_iou:
            is_best = True
            best_iou = iou_combined_mean

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': conf_segnet_model.state_dict(),
            'best_iou': best_iou,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, is_best, filename=opt.checkpointname)

