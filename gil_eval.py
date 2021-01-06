import numpy as np
import cv2

def eval_segmentation(src, tgt):
    s = (src / 255.0).astype(np.uint32)
    t = (tgt / 255.0).astype(np.uint32)

    s_abs = s.sum()
    t_abs = t.sum()

    s_int_t = np.bitwise_and(s, t)
    s_uni_t = np.bitwise_or(s, t)

    s_diff_t = s - s_int_t
    t_diff_s = t - s_int_t

    #
    # cv2.imshow("s_int_t", s_int_t.astype(np.uint8) * 255)
    # cv2.imshow("s_uni_t", s_uni_t.astype(np.uint8) * 255)
    # cv2.imshow("s_diff_t", s_diff_t.astype(np.uint8) * 255)
    # cv2.imshow("t_diff_s", t_diff_s.astype(np.uint8) * 255)

    overlab = s_int_t.sum() / t.sum()
    jaccard = s_int_t.sum() / s_uni_t.sum()
    dice = 2.0 * s_int_t.sum() / (s.sum() + t.sum())
    fn = t_diff_s.sum() / t.sum()
    fp = s_diff_t.sum() / s.sum()

    return overlab, jaccard, dice, fn, fp

def eval_total_overlab(src, tgt):

    return


def eval_Jaccard(src, tgt):
    return


def eval_Dice(src, tgt):
    return


def eval_false_negative(src, tgt):
    return


def eval_false_positive(src, tgt):
    return
