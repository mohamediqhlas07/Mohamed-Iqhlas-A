import numpy as np
def single_to_multilabel(class_index, num_classes):
    """
    Converts single label to multi-label binary vector
    """
    label = np.zeros(num_classes, dtype=np.float32)
    label[class_index] = 1.0
    return label