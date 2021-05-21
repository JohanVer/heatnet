import torch
from models import conf_segnet
import numpy as np
import thermal_loader
from yaml import load
from torch import nn
from utils import getPaths
import argparse
import cv2
from torch.autograd import Variable
from vis_utils import ColorCode, visImage3Chan, visDepth
from utils import calculate_ious


def validate_model(model, val_loader, modalities, mode="day", vis=False, save_dir=""):

    print('Evaluating {}.'.format(mode))

    preds = Variable(torch.zeros(len(val_loader), 320, 704))
    gts = Variable(torch.zeros(len(val_loader), 320, 704))

    preds_rgb = Variable(torch.zeros(len(val_loader), 3, 320, 704))
    gts_rgb = Variable(torch.zeros(len(val_loader), 3, 320, 704))
    imgs_rgb = Variable(torch.zeros(len(val_loader), 3, 320, 704))
    imgs_ir = Variable(torch.zeros(len(val_loader), 320, 704))

    color_coder = ColorCode(256)

    for i, batch in enumerate(val_loader):
        print('Validating ... %d of %d ...' % (i, len(val_loader)))
        rgb_im = batch['rgb'].cuda()
        ir_im = batch['ir'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        # encoder

        if 'rgb' in modalities and 'ir' in modalities:
            in_night = [rgb_im, ir_im]
        elif 'rgb' in modalities:
            in_night = [rgb_im]
        elif 'ir' in modalities:
            in_night = [ir_im]
        else:
            print('No known modality selected....')
            exit()

#        in_night = torch.cat([rgb_im, ir_im], dim=1)
#         in_night = torch.cat([in_night, in_night], dim=0)
        in_night_d = []
        for t in in_night:
            in_night_d.append(torch.cat([t, t], dim=0))

        with torch.no_grad():
            segmented, _, _ = model(*in_night_d)
        segmented = segmented[0:1, ...]

        if vis:
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label, argmax=False)
            visImage3Chan(batch['rgb_org'], 'RGB')
            cv2.imshow('Pred Seg', pred_color)
            cv2.imshow('GT Seg', gt_color)
            # visDepth(batch['ir_org'].clamp(0.3, 1.0), 'IR')
            visDepth(batch['ir_org'], 'IR')
            cv2.waitKey()

        if save_dir is not "":
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label, argmax=False)
            rgb_image = np.transpose(batch['rgb_org'].cpu().data.numpy().squeeze(), (1, 2, 0))

            # ir_cv = batch['ir_org'].clamp(0.2, 0.7).cpu().data.numpy().squeeze()
            ir_cv = batch['ir_org'].cpu().data.numpy().squeeze()
            cv2.normalize(ir_cv, ir_cv, 0, 255, cv2.NORM_MINMAX)
            ir_cv = cv2.applyColorMap(ir_cv.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(save_dir + '/pred_' + str(i) + ".png", (pred_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/gt_' + str(i) + ".png", (gt_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/rgb_' + str(i) + ".png",
                        cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(save_dir + '/ir_' + str(i) + ".png", ir_cv)

        segmented_argmax = torch.argmax(segmented.cpu(), 1)

        # account for offset in GT labels due to _background_ classes
        gts[i, :, :] = label.squeeze()
        preds[i, :, :] = segmented_argmax.squeeze()

        segmentation_gt = gts[i, :, :]
        segmentation_gt = color_coder.color_code_labels(segmentation_gt, argmax=False)
        segmentation_gt = np.transpose(segmentation_gt, (2, 0, 1))
        segmentation_gt = np.expand_dims(segmentation_gt, axis=0)
        segmentation_gt = torch.from_numpy(segmentation_gt).float()
        gts_rgb[i, :, :, :] = segmentation_gt

        segmentation_pred = preds[i, :, :]
        segmentation_pred = color_coder.color_code_labels(segmentation_pred, argmax=False)
        segmentation_pred = np.transpose(segmentation_pred, (2, 0, 1))
        segmentation_pred = np.expand_dims(segmentation_pred, axis=0)
        segmentation_pred = torch.from_numpy(segmentation_pred).float()

        preds_rgb[i, :, :, :] = segmentation_pred
        imgs_rgb[i, :, :, :] = rgb_im.squeeze()
        imgs_ir[i, :, :] = ir_im.squeeze()

    ious = calculate_ious(preds, gts)

    return ious


def createValloader(data_dirnames):
    dataloader_val = thermal_loader.ThermalTestDataLoader(*getPaths(data_dirnames))

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1,
                                               pin_memory=True,
                                               drop_last=False)
    return val_loader


def initModelPartial(model, weights_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testroot_day', default='/data/zuern/datasets/thermal_seg/test_set_day/converted', type=str,
                        help='root directory of the daytime testing split')
    parser.add_argument('--testroot_night', default='/data/zuern/datasets/thermal_seg/test_set_night/converted', type=str,
                        help='root directory of the nighttime testing split')
    parser.add_argument('--weights', default='model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--yaml-config', default='config.yaml', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')

    args = parser.parse_args()
    opt = parser.parse_args()

    conf_segnet_model = conf_segnet.conv_segnet(pretrained=opt.pretraining['value'],
                                                disc_arch=opt.discarch['value'],
                                                num_critics=opt.num_critics['value'],
                                                feedback_seg=opt.feedback_seg['value'],
                                                no_conf=opt.no_conf['value'],
                                                modalities=opt.modalities['value'],
                                                input_adapter=opt.train_input_adapter['value'],
                                                cert_branch=opt.cert_branch['value'],
                                                arch=opt.arch['value'],
                                                late_fusion=opt.late_fusion['value'])

    # load checkpoint
    if args.cuda:
        conf_segnet_model.cuda()
        conf_segnet_model = nn.DataParallel(conf_segnet_model.cuda(), [0])
    checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage)
    conf_segnet_model.load_state_dict(checkpoint['state_dict'])

    # create dataloader
    val_loader_night = createValloader([args.testroot_night])
    val_loader_day = createValloader([args.testroot_day])

    ious_night = validate_model(conf_segnet_model.module.trgb_segnet, val_loader_night, opt.modalities['value'], mode="night")
    ious_day = validate_model(conf_segnet_model.module.trgb_segnet, val_loader_day, opt.modalities['value'], mode="day")

    # calculate simple mean between daytime and nighttime accuracies
    ious_combined = (ious_day + ious_night) / 2
    iou_combined_mean = np.mean(ious_combined)

    # log combined ious
    print_dict = {
        "mean IoU": iou_combined_mean,
        "IoU road,parking": ious_combined[0],
        "IoU ground,sidewalk": ious_combined[1],
        "IoU building,": ious_combined[2],
        'IoU curb': ious_combined[3],
        'IoU fence': ious_combined[4],
        'IoU pole,traffic light,traffic sign': ious_combined[5],
        'IoU vegetation': ious_combined[6],
        'IoU terrain': ious_combined[7],
        'IoU sky': ious_combined[8],
        'IoU person,rider': ious_combined[9],
        'IoU car,truck,bus,train': ious_combined[10],
        'IoU motorcycle,bicycle': ious_combined[11],
    }

    print(print_dict)


if __name__ == '__main__':
    main()