import argparse
from cfg import Config as cfg
import cv2
from recognizers import TextRecognizer
import caffe
from detectors import TextProposalDetector, TextDetector
from other import draw_boxes, rank_boxes, resize_im, refine_boxes, CaffeModel
from utils.timer import Timer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def init_models(args):
    if cfg.PLATFORM == "GPU":
        caffe.set_mode_gpu()
        caffe.set_device(cfg.TEST_GPU_ID)
    else:
        caffe.set_mode_cpu()
    # initialize the detectors
    text_proposals_detector=TextProposalDetector(CaffeModel(args.DET_NET_DEF_FILE, args.DET_MODEL_FILE))
    text_detector=TextDetector(text_proposals_detector)
    text_recognizer = TextRecognizer(args.RECOG_MODEL_FILE, cfg.TEXT_RECOG_ALPHABET)

    return text_detector, text_recognizer


def text_detect(args, text_detector, image_path):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%image_path

    im=cv2.imread(image_path)
    im_small, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)

    timer = Timer()
    timer.tic()
    text_lines=text_detector.detect(im_small)
    text_lines=text_lines / f # project back to size of original image
    text_lines = refine_boxes(im, text_lines)
    text_lines=rank_boxes(text_lines)
    print "Number of the detected text lines: %s" % len(text_lines)
    print "Detection Time: %f" % timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if args.SAVE_IMAGE_WITH_BOX:
        im_with_text_lines = draw_boxes(im, text_lines, is_display=False, caption=image_path, wait=False)
        if im_with_text_lines is not None:
            cv2.imwrite(image_path+'_boxes.jpg', im_with_text_lines)

    return im, text_lines


def text_recog(im, text_lines, text_recognizer):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s" % image_path
    timer = Timer()
    timer.tic()
    predictions = text_recognizer.predict(im, text_lines)
    print "Recognition Time: %f" %timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    return predictions


def save_text_lines(text_lines, image_path):
    text_file = image_path + ".txt"
    with open(text_file,'w') as f:
        for text in text_lines:
            f.write(text+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Test OCR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    parser.add_argument('--DET_NET_DEF_FILE', help='prototxt for text detection net',
                        default="models/deploy.prototxt", type=str)
    parser.add_argument('--DET_MODEL_FILE', help='caffemodel for text detection net',
                        default="models/ctpn_trained_model.caffemodel", type=str)
    parser.add_argument('--RECOG_MODEL_FILE', help='pytorch model for text recognition net',
                        default="models/netCRNN63.pth", type=str)
    parser.add_argument('--SAVE_IMAGE_WITH_BOX', help='save text detection result', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    text_detector, text_recognizer = init_models(args)
    if args.image is not None:
        image_path = args.image
        im, text_bboxes = text_detect(args, text_detector, image_path)
        predictions = text_recog(im, text_bboxes, text_recognizer)
        save_text_lines(predictions, image_path)
    else:
        with open(args.imagelist,'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            im, text_bboxes = text_detect(args, text_detector, image_path)
            predictions = text_recog(im, text_bboxes, text_recognizer)
            save_text_lines(predictions, image_path)
