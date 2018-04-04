# brain-tumor-seg
Implement V-net model for brain tumor segmentation.

## Install

First you need to install anaconda locally: http://anaconda.com

- go to your local directory: ```cd /home/your_sunet_id ```
- get anconda: ```wget xxxxxxxxxxx ```
(installing light anaconda is enough)
- create a virtual environment: ```conda install xxx```

Then you need to install python packages:
```
pip install -r requirements.txt
```

In order to use tensorflow with GPU, you need to launch the CUDA module. I recommend adding these lines to your bash_profile
(for instance you can edit with vim: vim ~/.bash_profile):
```
module load CUDA
xxxxxxx
```
(don't forget to run 'source ~/.bahs_profile' for your changes to be taken into account)

## Data

The data is stored on crosswood at '/labs/gevaertlab/xxxxxx'. You will download it from this path and create the datasets on the GPU cluster on SSD memory for faster access:
```
python utils/prep_brats.py
python utils/prep_rembrandt.py

```
This will create a new drectory '/local_scratch/your_sunet_id_scratch' with folders for each datasets. This will also automatically preprocess the data.


## Train
You can train a model by running the command:
```
python fcn_train --cfg-path=path_to_your_config_file
```

For example: python fcn_train --cfg-path=config_files/fcn_train_concat_2017_v34.cfg

In order not to use all the GPUs on the cluster, I recommend running:

```
CUDA_VISIBLE_DECIVES=Xxx python fcn_train --cfg-path=path_to_your_config_file
```
where 'Xxx' is any id of GPU (here Xxx = 0, 1, 2, 3)

For example: CUDA_VISIBLE_DECIVES=1 python fcn_train --cfg-path=config_files/fcn_train_concat_2017_v34.cfg

## Test
To have the details results on HGG and LGG patients for one particular model you can run

```
CUDA_VISIBLE_DECIVES=Xxx python fcn_test --cfg-path=path_to_your_config_file
```

For example: CUDA_VISIBLE_DECIVES=1 python fcn_test --cfg-path=config_files/fcn_train_concat_2017_v34.cfg

## Visualize results

You can use the jupyter notebook 'mri_viewer.ipynb' to visualize the results of segmentation of your model.

First you need to launch jupyter notebook on the GPU cluster. For that, you can open a tmux window and run:
```
CUDA_VISIBLE_DEVICES=Xxx jupyter notebook -xxxx -xxx -xxx
```

For example: 

Then you need to create a SSH tunnel to use jupyter notebook locally on your computer:
```
ssh -N -f -L localhost:local_port:localhost:server_port your_sunet_id@bmir-ct-gpu-5.stanford.edu
```
where 'local_port' is the port you want to use locally, and 'server_port' is the port you listen to on the server (= the port on which you launched jupyter notebook in the previous step)

For example:  ssh -N -f -L localhost:8888:localhost:8889 romains@bmir-ct-gpu-5.stanford.edu

Now you can navigate to http://localhost:local_port in an internet browser and use jupyter notebook normally (note that you might be prompted to enter a token. In that case you should look in your tmux window where you jupyter notebbok is running and copy paste the token after '.....token_id=xxxxxxxxxxxx'.)

You should open the notebook 'mri_viewer.ipynb' and follow the instructions inside to visualize the results of the segmentation for various models on various patients.