# Setup for visualizing RTOG images from the Gevaert lab

Please first refer to the **First-time setup** section. 



### Everyday use

Connecting to the server and running distant Jupyter instance *(requires the Stanford VPN; will ask for your password )*

```bash
ssh USERNAME@bmir-ct-gpu-5.stanford.edu
cd romain
nb_server
```



On a new terminal, connecting to the server *(requires the Stanford VPN; will ask for your password )*

```bash
nb_bridge
```



Connect to the Jupyter server on your internet browser (Safari, Chrome, Firefox), by following the link [localhost:8800/](localhost:8800/).



Navigate to `rtog/radiologist_classification/mri_viewer-rtog.ipynb`



### First-time setup

Setting local shortcuts, replacing *USERNAME*

```bash
touch .bash_profile
echo 'alias nb_bridge='\''echo "ssh -N -L localhost:8800:localhost:8900 USERNAME@bmir-ct-gpu-5.stanford.edu"; ssh -N -L localhost:8800:localhost:8999 USERNAME@bmir-ct-gpu-5.stanford.edu'\''' >> .bash_profile
source .bash_profile
```



Connecting to the server  *(requires the Stanford VPN; will ask for your password )*

```bash
ssh USERNAME@bmir-ct-gpu-5.stanford.edu
```



Installing python

`````````bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
`````````



Downloading project codebase *(will ask for my password)*

```bash
git clone https://marcthib@bitbucket.org/gevaertlab/romain.git
```



Setting local shortcuts, replacing *USERNAME*

```bash
touch .bash_profile
echo 'alias nb_server="jupyter notebook --no-browser --ip=0.0.0.0 --port=8999"' >> .bash_profile
echo 'export PYTHONPATH="${PYTHONPATH}:/home/USERNAME/romain"' >> .bash_profile
source .bash_profile
```



Installing required packages

```bash
conda install -c conda-forge pydicom

while read requirement; do conda install --yes $requirement; done < romain/rtog/radiologist_classification/requirements.txt
```

