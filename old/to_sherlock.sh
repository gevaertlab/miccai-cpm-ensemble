#!/bin/bash

src_path=~/Dropbox/Workspace/tumor-seg
dst_path=rsub@sherlock.stanford.edu:\~/tumor-seg

cd "$src_path"

# scp train.py "$dst_path"
# scp prime.py "$dst_path"
# scp eval.py "$dst_path"
# scp get_probs.py "$dst_path"
# scp train_hb.py "$dst_path"
scp train_only.py "$dst_path"

scp models/model.py "$dst_path"/models
scp models/baseline.py "$dst_path"/models
# scp models/heidelberg.py "$dst_path"/models
scp models/baseline_smaller.py "$dst_path"/models

scp utils/config.py "$dst_path"/utils
# scp utils/config_hb.py "$dst_path"/utils
scp utils/data_utils.py "$dst_path"/utils
scp utils/data_iterator.py "$dst_path"/utils
scp utils/dice_score.py "$dst_path"/utils
