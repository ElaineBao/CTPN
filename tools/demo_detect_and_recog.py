import argparse
from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import cv2, caffe
from detectors import TextProposalDetector, TextDetector
from recognizers import TextRecognizer
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
    text_recognizer=TextRecognizer(args.RECOG_MODEL_FILE,cfg.TEXT_RECOG_ALPHABET)

    return text_detector, text_recognizer


def text_predict(args, text_detector, text_recognizer):
    timer=Timer()
    im_file = args.image
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%im_file

    im=cv2.imread(im_file)

    timer.tic()

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)
    print "Number of the detected text lines: %s" % len(text_lines)
    if args.SAVE_IMAGE_WITH_BOX:
        im_with_text_lines = draw_boxes(im, text_lines, is_display=False, caption=im_file, wait=False)
        if im_with_text_lines is not None:
            cv2.imwrite(im_file+'_boxes.jpg', im_with_text_lines)

    predictions=text_recognizer.predict(im, text_lines)
    print predictions

    print "Time: %f"%timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Done."


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', type=str)
    parser.add_argument('--DET_NET_DEF_FILE', help='prototxt for text detection net',
                        default="models/deploy.prototxt", type=str)
    parser.add_argument('--DET_MODEL_FILE', help='caffemodel for text detection net',
                        default="models/ctpn_trained_model.caffemodel", type=str)
    parser.add_argument('--RECOG_MODEL_FILE', help='pytorch model for text recognition net',
                        default="models/idcard_addr_10.pth", type=str)
    parser.add_argument('--SAVE_IMAGE_WITH_BOX', help='save text detection result', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    text_detector, text_recognizer = init_models(args)
    text_predict(args, text_detector, text_recognizer)