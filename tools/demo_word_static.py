import argparse
from jieba import posseg as psg
import operator
import io


def load_stop_words(path="src/stop_words.utf8"):
    stop_words = set()
    with io.open(path, 'r', encoding='utf8') as f:
        for l in f:
            l = l.strip()
            stop_words.add(l)
    return stop_words


class WordAnalyzer:
    def __init__(self):
        self.stop_words = load_stop_words()

    def cut_words(self, words):
        words = psg.cut(words)
        return [w for w, t in words if w not in self.stop_words]

    def process(self, file_name):
        """

        :param file_name: file path
        :param top_k: if top_k=-1 print all, else top k
        :return: tuple
        """
        article = ""
        with io.open(file_name, encoding='utf8') as f:
            for line in f:
                article += line.strip('\n')

        words = self.cut_words(article)

        word_statics = {}

        for word in words:
            counts = word_statics.get(word, 0)
            word_statics[word] = counts + 1

        sorted_words = sorted(word_statics.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_words

    def save_plt(self, word_count, image_path, top_k=-1):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pylab import mpl

        if top_k > 0:
            word_count = word_count[:top_k]
        mpl.rcParams['font.sans-serif'] = ['FongSong']
        mpl.rcParams['axes.unicode_minus'] = False

        fig = plt.figure()
        label = list(map(lambda x: x[0], word_count))
        value = list(map(lambda y: y[1], word_count))
        plt.bar(range(len(value)), value, tick_label=label)
        #plt.show()
        plt.savefig(image_path+'_word_statics.jpg')
        plt.title('word static')
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Test OCR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    parser.add_argument('--top_k', help='save top_k words in the word static image', default=-1, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    analyzer = WordAnalyzer()

    if args.image is not None:
        image_path = args.image
        text_path = image_path + '.txt'
        words_tuple = analyzer.process(text_path)
        analyzer.save_plt(words_tuple, image_path, args.top_k)
    else:
        with open(args.imagelist, 'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            text_path = image_path + '.txt'
            words_tuple = analyzer.process(text_path)
            analyzer.save_plt(words_tuple, image_path, args.top_k)