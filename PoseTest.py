import torch
from torchvision import transforms

import libs.yolov7.utils.datasets as datasets
import libs.yolov7.utils.general as general
import libs.yolov7.utils.plots as plots

letterbox = datasets.letterbox
non_max_suppression_kpt = general.non_max_suppression_kpt
output_to_keypoint, plot_skeleton_kpts = plots.output_to_keypoint, plots.plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np


def load_model():
    device = torch.device("cpu")
    model = torch.load('libs/yolov7/yolov7-w6-pose.pt', map_location=device)['model']
    # put in inference mode
    model.float().eval()

    return model


def run_inference(cap, model):
    image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, 960, stride=64, auto=True)
    image = transforms.ToTensor()(image)

    image = image.unsqueeze(0)
    output, _ = model(image)
    return output, image


def visualize_output(output, image, model):
    output = non_max_suppression_kpt(output,
                                     0.25,
                                     0.65,
                                     nc=model.yaml['nc'],
                                     nkpt=model.yaml['nkpt'],
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.show()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cap not success")
        return
    model = load_model()
    output, image = run_inference(cap, model)
    visualize_output(output, image, model)



if __name__ == '__main__' :
    main()
