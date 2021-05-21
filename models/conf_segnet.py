import torch
import torch.nn as nn

from models import discriminator
import utils
from models import trgb_segnet as models
from utils import weights_init_normal
from models import critic_resnet
from models import downscale_network
from models import input_adapter as input_adapter_model
from models import build_net


def create_critic(disc_arch, input_num):
    if disc_arch == 'cyclegan':
            return discriminator.FCDiscriminator(input_num)

    elif 'resnet' in disc_arch:
            c = getattr(critic_resnet, disc_arch)(False, False, **{'num_classes': 1, 'input_maps': input_num})
            c.apply(weights_init_normal)
            return c

class conv_segnet(nn.Module):
    def __init__(self, pretrained=True, disc_arch='resnet', num_critics=6, feedback_seg=False, no_conf=False, modalities='ir_rgb', input_adapter=False, cert_branch=False, arch='custom', late_fusion=False):
        super(conv_segnet, self).__init__()

        num_input_channels = 0
        if 'rgb' in modalities:
            num_input_channels += 3
            print('Using RGB')

        if 'ir' in modalities:
            num_input_channels += 1
            print('Using IR')

        print('Total numbers of input channels: %d' % (num_input_channels))

        if arch == 'custom':
            self.trgb_segnet = models.ResNeXt(**{"structure": [3, 4, 6, 3], "input_channels": num_input_channels, "cert_branch": cert_branch, "late_fusion": late_fusion})
            if late_fusion:
                critic_num = [13, 768, 1024, 512, 256*2, 64*2]
            else:
                critic_num = [13, 512, 1024, 512, 256, 64]
        elif arch =='pspnet':
            self.trgb_segnet = build_net.build_network(None, 'resnet50', in_channels=num_input_channels, late_fusion=late_fusion)
            if late_fusion:
                critic_num = [13, 2048, 1024, 512*2, 256*2, 64*2]
                print('Activated late fusion ...')
            else:
                critic_num = [13, 2048, 1024, 512, 256, 64]
        else:
            print('Not supported model arch: %s' % (arch))

        self.trgb_segnet.apply(weights_init_normal)
        self.feedback_seg = feedback_seg
        self.input_adapter = input_adapter

        if input_adapter:
            self.input_adapter_net = input_adapter_model.UNet(num_input_channels, num_input_channels)
            self.adapter_disc = create_critic(disc_arch, num_input_channels)

        if not no_conf:
            if feedback_seg:
                num_downscale = [3, 3, 3, 2, 2]
                self.downscale_nets = torch.nn.ModuleList()
                for i in range(1, len(critic_num)):
                    critic_num[i] = critic_num[i] + 12

                # Models for downsizing segmentation output:
                for i in range(len(num_downscale)):
                    self.downscale_nets.append(downscale_network.DownNet(num_downscale[i]))

            critic_num = critic_num[0:num_critics]
            self.critics = torch.nn.ModuleList()

            print('Creating %d critics....' % (len(critic_num)))

            for i in range(len(critic_num)):
                self.critics.append(create_critic(disc_arch, critic_num[i]))

        if pretrained:
            utils.initModelRenamed(self.trgb_segnet, 'models_finished/training_nc_irrgb_best.pth', 'module.', '')

        self.phase = "train_seg"
        self.no_conf = no_conf

    def setLearningModel(self, module, val):
        for p in module.parameters():
            p.requires_grad = val

    def setPhase(self, phase):
        self.phase = phase
        print("Switching to phase: %s" % self.phase)
        if self.phase == "train_seg":
            # self.trgb_segnet.setForwardDecoder(True)
            if not self.no_conf:
                for c in self.critics:
                    self.setLearningModel(c, False)
            self.setLearningModel(self.trgb_segnet, True)
        elif self.phase == "train_critic":
            # self.trgb_segnet.setForwardDecoder(True)
            if not self.no_conf:
                for c in self.critics:
                    self.setLearningModel(c, True)
            self.setLearningModel(self.trgb_segnet, False)

    def forward(self, input_a, input_b):
        output = {}
        if self.input_adapter:
            input_a = self.input_adapter_net(input_a)
            input_b = self.input_adapter_net(input_b)
            output['input_a'] = input_a
            output['input_b'] = input_b

        pred_label_day, inter_f_a, cert_a = self.trgb_segnet(*input_a)
        pred_label_night, inter_f_b, cert_b = self.trgb_segnet(*input_b)

        if not self.no_conf:
            output['critics_a'] = []
            output['critics_b'] = []

            for i, c in enumerate(self.critics):
                if self.feedback_seg:
                    if i > 0:
                        inter_f_a[i] = torch.cat([inter_f_a[i], self.downscale_nets[i-1](pred_label_day)], dim=1)
                        inter_f_b[i] = torch.cat([inter_f_b[i], self.downscale_nets[i-1](pred_label_night)], dim=1)

                output['critics_a'].append(c(inter_f_a[i]))
                output['critics_b'].append(c(inter_f_b[i]))

            if self.input_adapter:
                output['critics_a'].append(self.adapter_disc(input_a))
                output['critics_b'].append(self.adapter_disc(input_b))

        output['pred_label_a'] = pred_label_day
        output['pred_label_b'] = pred_label_night
        output['cert_a'] = cert_a
        output['cert_b'] = cert_b
        output['inter_f_b'] = inter_f_b

        return output
