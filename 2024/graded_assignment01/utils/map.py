import numpy as np

from .phos_generator import gen_phos_label
from .phoc_generator import gen_phoc_label


# from: https://github.com/anuj-rai-23/PHOSC-Zero-Shot-Word-Recognition
def get_map_dict(x):
    phos_labels = gen_phos_label(x)
    phoc_labels = gen_phoc_label(x)
    test_labels = dict()
    for x in phos_labels:
        test_labels[x] = np.concatenate((phos_labels[x], phoc_labels[x]), axis=0)
    return test_labels


if __name__ == '__main__':
    print(get_map_dict(['hello', 'hi']))
