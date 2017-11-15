import argparse
from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import cv2, caffe
from detectors import TextProposalDetector, TextDetector
from utils.timer import Timer


def init_models(args):
    if cfg.PLATFORM == "GPU":
        caffe.set_mode_gpu()
        caffe.set_device(cfg.TEST_GPU_ID)
    else:
        caffe.set_mode_cpu()

    # initialize the detectors
    text_proposals_detector=TextProposalDetector(CaffeModel(args.DET_NET_DEF_FILE, args.DET_MODEL_FILE))
    text_detector=TextDetector(text_proposals_detector)

    return text_detector

def text_detect(args, text_detector, image_path):

    timer=Timer()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%image_path

    im=cv2.imread(image_path)
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)

    timer = Timer()
    text_lines=text_detector.detect(im)
    print "Number of the detected text lines: %s" % len(text_lines)
    print "Detection Time: %f" % timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if args.SAVE_IMAGE_WITH_BOX:
        im_with_text_lines = draw_boxes(im, text_lines, is_display=False, caption=image_path, wait=False)
        if im_with_text_lines is not None:
            cv2.imwrite(image_path+'_boxes.jpg', im_with_text_lines)

    return im, text_lines


def save_text_bboxes(text_lines, image_path):
    text_file = image_path + "_bboxes.txt"
    with open(text_file,'w') as f:
        for rect in text_lines:
            for coord in rect:
                f.write('%f\t'%coord)
            f.write('\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Test OCR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    parser.add_argument('--DET_NET_DEF_FILE', help='prototxt for text detection net',
                        default="models/deploy.prototxt", type=str)
    parser.add_argument('--DET_MODEL_FILE', help='caffemodel for text detection net',
                        default="models/ctpn_trained_model.caffemodel", type=str)
    parser.add_argument('--SAVE_IMAGE_WITH_BOX', help='save text detection result', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    text_detector = init_models(args)
    if args.image is not None:
        image_path = args.image
        im, text_bboxes = text_detect(args, text_detector, image_path)
        if text_bboxes is not None:
            save_text_bboxes(text_bboxes, image_path)
    else:
        with open(args.imagelist,'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            im, text_bboxes = text_detect(args, text_detector, image_path)
            if text_bboxes is not None:
                save_text_bboxes(text_bboxes, image_path)
