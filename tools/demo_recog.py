import argparse
from cfg import Config as cfg
import cv2
from utils.timer import Timer
from recognizers import TextRecognizer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def init_models(args):
    text_recognizer = TextRecognizer(args.RECOG_MODEL_FILE, cfg.TEXT_RECOG_ALPHABET)
    return text_recognizer

def text_recog(args, text_recognizer, text_lines, image_path):
    timer = Timer()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s" % image_path

    im = cv2.imread(image_path)

    timer.tic()
    predictions = text_recognizer.predict(im, text_lines)
    print "Recognition Time: %f" %timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    return predictions


def load_text_bboxes(image_path):
    text_lines = []
    text_file = image_path + "_bboxes.txt"
    with open(text_file, 'r') as f:
        for line in f:
            lst = line.strip().split('\t')
            rect = [int(float(i)) for i in lst]
            text_lines.append(rect)
    return text_lines


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
    parser.add_argument('--RECOG_MODEL_FILE', help='pytorch model for text recognition net',
                        default="models/netCRNN63.pth", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    text_recognizer = init_models(args)
    if args.image is not None:
        image_path = args.image
        text_bboxes = load_text_bboxes(image_path)
        predictions = text_recog(args, text_recognizer, text_bboxes, image_path)
        save_text_lines(predictions,image_path)
    else:
        with open(args.imagelist, 'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            text_bboxes = load_text_bboxes(image_path)
            predictions = text_recog(args, text_recognizer, text_bboxes, image_path)
            save_text_lines(predictions, image_path)
