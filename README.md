# brain-tumor-seg
Implement V-net model for brain tumor segmentation.


## Install

First you need to install anaconda locally: https://conda.io/docs/user-guide/install/linux.html
(installing miniconda is enough).
Then you create a virtual environment with Python 3.

Now you can install the required python packages:
```
pip install -r requirements.txt
```

In order to use tensorflow with GPU, you need to launch the CUDA module everytime you connect to the server. I recommend adding this line to your bash_profile (for instance you can edit with vim: vim ~/.bash_profile):
```
module load cuda/9.0
```
(we load cuda 9.0 because cuda 9.1 is not compatible with tensorflow 1.5.0 as of today)
Don't forget to run ```source ~/.bash_profile``` for your changes to be effective.


## Data

The data is stored on crosswood at '/labs/gevaertlab/data/tumor_segmentation/'. You will import it from this path and create the datasets on the GPU cluster on SSD memory for faster access:
```
python utils/prep_brats2017_data.py
python utils/prep_rembrandt_data.py

```
This will create a new drectory '/local_scratch/your_sunet_id_scratch' with folders for each datasets and split the data into a training set and a test set.


## Train
You can train a model by running the command:
```
python fcn_train.py --cfg-path=path_to_your_config_file
```
For example: python fcn_trai.pyn --cfg-path=config_files/fcn_train_concat_2017_v34.cfg

In order not to use all the GPUs on the cluster, I recommend running:
```
CUDA_VISIBLE_DECIVES=<gpu_id> python fcn_train.py --cfg-path=path_to_your_config_file
```
where 'gpu_id' is any id of GPU (here gpu_id = 0, 1, 2, 3).
For example: CUDA_VISIBLE_DEVICES=0 python fcn_train.py --cfg-path=config_files/fcn_train_concat_2017_v34.cfg


## Test
To have the details results on HGG and LGG patients for one particular model you can run

```
CUDA_VISIBLE_DECIVES=<gpu_id> python fcn_test.py --cfg-path=path_to_your_config_file
```
For example: CUDA_VISIBLE_DEVICES=0 python fcn_test --cfg-path=config_files/fcn_train_concat_2017_v34.cfg


## Visualize results

You can use the jupyter notebook 'mri_viewer.ipynb' to visualize the results of segmentation of your model.

First you need to launch jupyter notebook on the GPU cluster. For that, you can open a tmux window and run:
```
CUDA_VISIBLE_DEVICES=<gpu_id> jupyter notebook --no-browser --ip=0.0.0.0 --port=<server_port>
```
For example: CUDA_VISIBLE_DEVICES=0 jupyter notebook --no-browser --ip=0.0.0.0 --port=8889

Then you need to create a SSH tunnel to use jupyter notebook locally on your computer:
```
ssh -N -f -L localhost:<local_port>:localhost:<server_port> your_sunet_id@bmir-ct-gpu-5.stanford.edu
```
where 'local_port' is the port you want to use locally, and 'server_port' is the port you listen to on the server (the same port as the one you used to launch jupyter notebook in the previous step).
For example:  ssh -N -f -L localhost:8888:localhost:8889 romains@bmir-ct-gpu-5.stanford.edu

Now you can navigate to http://localhost:8888 (if you use 8888 as local port) in an internet browser and use jupyter notebook normally (note that you might be prompted to enter a token. In that case you should look in your tmux window where you jupyter notebbok is running and copy paste the token from the line: http://0.0.0.0:8889/?token=TOKEN_TO_COPY')

You should open the notebook 'mri_viewer.ipynb' and follow the instructions inside to visualize the results of the segmentation for various models on various patients.