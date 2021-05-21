from random import randint
import numpy as np
import cv2
from c_relabeller.relabeller import relabel_image


class ColorCode():

    def get_coding_1(self):
        coding = {}

        coding[0] = [70, 70, 70]
        coding[1] = [244, 35, 232]
        coding[2] = [128, 64, 128]
        coding[3] = [168, 168, 168]
        coding[4] = [0, 255, 255]
        coding[5] = [255, 165, 0]
        coding[6] = [107, 142, 35]
        coding[7] = [255, 255, 0]
        coding[8] = [70, 130, 180]
        coding[9] = [220, 20, 60]
        coding[10] = [0, 255, 0]
        coding[11] = [190, 153, 153]

        coding[12] = [0, 0, 0]
        coding[13] = [0, 0, 0]

        for k, v in coding.items():
            v = np.flip(v)
            coding[k] = v

        return coding

    def color_code_labels(self, net_out, argmax=True):
        if argmax:
            labels, indices = net_out.max(1)
            labels_cv = indices.cpu().numpy().squeeze()
        else:
            labels_cv = net_out.cpu().numpy().squeeze()

        color_coded = np.asarray(relabel_image(labels_cv.astype(np.uint8), self.color_coding))

        return color_coded / 255.

    def __init__(self, max_classes):
        super(ColorCode, self).__init__()
        self.color_coding = self.get_coding_1()


def visImage3Chan(data, name):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, cv)
    return cv


def visImage1Chan(data, name):
    cv = data.cpu().data.numpy().squeeze()
    cv2.normalize(cv, cv, 0, 255, cv2.NORM_MINMAX)
    cv = cv.astype(np.uint8)
    cv2.imshow(name, cv)
    return cv

def visDepth(data, name, normalize=True):
    disp_cv = data.cpu().data.numpy().squeeze()
    if normalize:
        cv2.normalize(disp_cv, disp_cv, 0, 255, cv2.NORM_MINMAX)
    else:
        disp_cv *= 255
    disp_cv_color = cv2.applyColorMap(disp_cv.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imshow(name, disp_cv_color)
    return disp_cv_color

def drawCriticsLabels(image, critic_vals, size_dots=10):
    num_critics = len(critic_vals)
    total_radius = num_critics * size_dots
    disp_cv_color = cv2.circle(image,
                               (image.shape[1] - (total_radius + 1), image.shape[0] - (total_radius + 1)),
                               total_radius + 1, (255, 255, 255), -1)
    size_circle = total_radius / num_critics
    for i, c in enumerate(critic_vals):
        disp_cv_color = cv2.circle(disp_cv_color, (
        (image.shape[1] - (total_radius + 1)), int(image.shape[0] - (size_circle + int(i * 2 * size_circle)))),
                                   int(size_circle),
                                   (0, 255, 0) if c else (0, 0, 255), -1)

    return image


def visSegDisc(data, name, disc_class, vis=True):
    color_coder = ColorCode(13)
    disp_cv_color = color_coder.color_code_labels(data, False)

    drawCriticsLabels(disp_cv_color, disc_class)

    if vis:
        cv2.imshow(name, disp_cv_color)
    return disp_cv_color
