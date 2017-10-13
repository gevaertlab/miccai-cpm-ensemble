from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk

# helper functions
def im_path_to_arr(im_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(im_path))

def dice_score(x, y):
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    num = np.sum(np.logical_and(x_bool, y_bool)) * 2.
    denom = np.sum(x_bool) + np.sum(y_bool)
    return num/denom

def jaccard_score(x, y):
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    num = np.sum(np.logical_and(x_bool, y_bool)) * 1.
    denom = np.sum(np.logical_or(x_bool, y_bool))
    return num/denom

def overlap_score(x, y):
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    num = np.sum(np.logical_and(x_bool, y_bool)) * 1.
    denom = min(np.sum(x_bool), np.sum(y_bool))
    return num/denom

def entropy(x):
    return np.where((x > 0) & (x < 1),
           -x*np.log2(x) - (1-x)*np.log2(1-x),
           0)

################################################################################
# PROBS
################################################################################

# load data
_dict = np.load('get_probs_results.npz')
fprobs = _dict['fprobs']
val_ex_paths = _dict['val_ex_paths']

# compute dice scores
dices = []
jaccards = []
overlaps = []
for i, path in enumerate(val_ex_paths):
    name = path.split('/')[-1]
    local_path = os.path.join('../../data/brats_hgg_unnorm/', name)
    y = im_path_to_arr(os.path.join(local_path, 'tumor.mha'))
    pred = np.argmax(fprobs[i, ...], axis=3)
    dices.append(dice_score(y, pred))
    jaccards.append(jaccard_score(y, pred))
    overlaps.append(overlap_score(y, pred))

print(np.mean(dices), np.mean(jaccards), np.mean(overlaps))

# # # get images with worst and best dice score
# # worst_idx = np.argmin(dices)
# # best_idx = np.argmax(dices)
# # worst_prob = fprobs[worst_idx, ...]
# # best_prob = fprobs[best_idx, ...]

# # # cache useful data
# # np.savez('stats.npz',
# #          dices=dices,
# #          worst_prob=worst_prob,
# #          best_prob=best_prob) 

# # load useful data
# _dict = np.load('stats.npz')
# # dices = _dict['dices']
# # worst_idx = np.argmin(dices)
# # best_idx = np.argmin(dices)
# worst_prob = _dict['worst_prob']
# best_prob = _dict['best_prob']

# # get image and ground truth
# # path = val_ex_paths[worst_idx]
# # name = path.split('/')[-1]
# name = 'brats_tcia_pat370_1126'
# local_path = os.path.join('../../data/brats_hgg_unnorm/', name)
# worst_im = im_path_to_arr(os.path.join(local_path, 't1c.mha'))
# worst_gt = im_path_to_arr(os.path.join(local_path, 'tumor.mha'))

# # path = val_ex_paths[best_idx]
# # name = path.split('/')[-1]
# name = 'brats_tcia_pat361_0001'
# local_path = os.path.join('../../data/brats_hgg_unnorm/', name)
# best_im = im_path_to_arr(os.path.join(local_path, 't1c.mha'))
# best_gt = im_path_to_arr(os.path.join(local_path, 'tumor.mha'))

# worst_im_sl = worst_im[50, ...]
# best_im_sl = best_im[85, ...]
# worst_gt_sl = worst_gt[50, ...]
# best_gt_sl = best_gt[85, ...]

# # get pred
# worst_pred = np.argmax(worst_prob, axis=3)
# best_pred = np.argmax(best_prob, axis=3)
# worst_pred_sl = worst_pred[50, ...]
# best_pred_sl = best_pred[85, ...]

# # compute entropy
# worst_ent = entropy(worst_prob[..., 0])
# best_ent = entropy(best_prob[..., 0])
# worst_ent_sl = worst_ent[50, ...]
# best_ent_sl = best_ent[85, ...]

# # plot things
# fig = plt.figure()
# ax = plt.subplot(221)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('T1 Contrast-Enhanced')
# ax.imshow(best_im_sl, cmap='gray', interpolation=None)
# ax = plt.subplot(222)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Ground Truth Segmentation')
# ax.imshow(best_im_sl, cmap='gray', interpolation=None)
# ax.imshow(best_gt_sl, cmap='hot', vmin=0, vmax=1, alpha=0.5, interpolation=None)
# ax = plt.subplot(223)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Predicted Segmentation')
# ax.imshow(best_im_sl, cmap='gray', interpolation=None)
# ax.imshow(best_pred_sl, cmap='hot', vmin=0, vmax=1, alpha=0.5, interpolation=None)
# ax = plt.subplot(224)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Entropy of Prediction')
# ax.imshow(best_im_sl, cmap='gray', interpolation=None)
# ax.imshow(best_ent_sl, cmap='hot', vmin=0, vmax=1, alpha=0.5, interpolation=None)
# dice = dice_score(best_gt, best_pred)
# plt.suptitle('Dice Score = {}'.format(dice))
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.savefig('best_fig.png')

# fig = plt.figure()
# ax = plt.subplot(111)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('T1 Contrast-Enhanced')
# ax.imshow(best_im_sl, cmap='gray', interpolation=None)
# plt.savefig('best_t1c.png')

# fig = plt.figure()
# ax = plt.subplot(221)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('T1 Contrast-Enhanced')
# ax.imshow(worst_im_sl, cmap='gray', interpolation=None)
# ax = plt.subplot(222)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Ground Truth Segmentation')
# ax.imshow(worst_im_sl, cmap='gray', interpolation=None)
# ax.imshow(worst_gt_sl, cmap='hot', vmin=0, vmax=1, alpha=0.5, interpolation=None)
# ax = plt.subplot(223)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Predicted Segmentation')
# ax.imshow(worst_im_sl, cmap='gray', interpolation=None)
# ax.imshow(worst_pred_sl, cmap='hot', vmin=0, vmax=1, alpha=0.5, interpolation=None)
# ax = plt.subplot(224)
# ax.grid(False)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Entropy of Prediction')
# ax.imshow(worst_im_sl, cmap='gray', interpolation=None)
# ax.imshow(worst_ent_sl, cmap='hot', vmin=0, vmax=1, alpha=0.5, interpolation=None)
# dice = dice_score(worst_gt, worst_pred)
# plt.suptitle('Dice Score = {}'.format(dice))
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.savefig('worst_fig.png')

# ################################################################################
# # XVAL
# ################################################################################

# dices = []

# for i in range(4):
#     _dict = np.load('xval_eval_' + str(i) + '_results.npz')
#     dices.extend(_dict['val_fdices'])

# fig = plt.figure()
# ax = plt.subplot(111)
# ax.hist(dices, 20, rwidth=0.8)
# ax.set_xlabel('Dice Score')
# ax.set_ylabel('Frequency')
# ax.set_title('Histogram of Dice Scores')
# plt.savefig('full_hist.png')
