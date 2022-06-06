import os
import numpy as np
from PIL import Image


NUM_OBJECTS = 79
NUM_LABELS = NUM_OBJECTS + 3


def get_split_label_files(split_dir, data_dir, split_name):
    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(data_dir, line.strip()) for line in f if line.strip()]
        label_files = [p + "_label_kinect.png" for p in prefix]
    return label_files


def get_labels(label_files, threshold=0.1):
    gt_labels = []
    noisy_labels = []
    for label_file in label_files:
        gt_label = np.array(Image.open(label_file))
        noisy_label = gt_label.copy()

        # add noise
        noise = np.random.randint(0, NUM_LABELS, gt_label.shape)
        mask = np.random.uniform(0, 1, gt_label.shape) < threshold
        noisy_label[mask] = noise[mask]

        gt_labels.append(gt_label)
        noisy_labels.append(noisy_label)

    return gt_labels, noisy_labels


def evaluate(gt_labels, pred_labels):
    intersection = np.zeros(NUM_OBJECTS)
    union = np.zeros(NUM_OBJECTS)
    nb_samples = np.zeros(NUM_OBJECTS)

    cnt = 0
    for gt_label, pred_label in zip(gt_labels, pred_labels):
        # intersection
        I = pred_label[(pred_label == gt_label)]
        I = np.histogram(I, bins=NUM_OBJECTS, range=[0, NUM_OBJECTS - 1])[0]

        # union
        total_num_pred = np.histogram(pred_label, bins=NUM_OBJECTS, range=[0, NUM_OBJECTS - 1])[0]
        total_num_gt = np.histogram(gt_label, bins=NUM_OBJECTS, range=[0, NUM_OBJECTS - 1])[0]
        U = total_num_pred + total_num_gt - I

        # update sum
        intersection += I
        union += U
        nb_samples += total_num_gt

        cnt += 1
        if cnt % 100 == 0:
            print(f"finish {cnt} / {len(gt_labels)}")
    
    class_Acc = (intersection / nb_samples) * 100.0
    mAcc = np.mean(class_Acc[nb_samples > 0])
    class_IoU = (intersection / union) * 100.0
    mIoU = np.mean(class_IoU[union > 0])
    return class_Acc, mAcc, class_IoU, mIoU


def main():
    split_dir = "../datasets/training_data/splits"
    data_dir = "../datasets/training_data/data"
    label_files = get_split_label_files(split_dir, data_dir, "val")
    gt_labels, noisy_labels = get_labels(label_files, threshold=0.01)

    # evaluate mean pixel accuracy and mean IoU
    class_Acc, mAcc, class_IoU, mIoU = evaluate(gt_labels, noisy_labels)
    print("class Acc =", class_Acc)
    print("mAcc =", mAcc)
    print("class IoU =", class_IoU)
    print("mIoU =", mIoU)


if __name__ == "__main__":
    main()