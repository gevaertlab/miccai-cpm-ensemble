# Setup for visualizing RTOG images from the Gevaert lab

Please first refer to the **First-time setup** section. 



## Everyday use

###### Unix (Linux/Max)
Connect to the server and run a distant Jupyter instance *(requires the Stanford VPN; will ask for your password )*

```bash
ssh USERNAME@bmir-ct-gpu-5.stanford.edu
```
```bash
cd romain
nb_server
```

On a new terminal, create a bridge to the server *(requires the Stanford VPN; will ask for your password )*

```bash
ssh -N -L localhost:8800:localhost:8999 USERNAME@bmir-ct-gpu-5.stanford.edu
```



###### Windows

Connect to the server and run a distant Jupyter instance *(requires the Stanford VPN; will ask for your password )*

```bash
plink.exe -ssh USERNAME@bmir-ct-gpu-5.stanford.edu
```
```bash
cd romain
nb_server
```

On a new command line, create a bridge to the server *(requires the Stanford VPN; will ask for your password )*

```
plink.exe -N -L localhost:8800:localhost:8999 USERNAME@bmir-ct-gpu-5.stanford.edu
```



###### (All) Launch the visualization UI 

Connect to the Jupyter server on your internet browser (Safari, Chrome, Firefox), by following the link [localhost:8800/](localhost:8800/).



Navigate to `rtog/radiologist_classification/mri_viewer-rtog.ipynb`



## First-time setup

###### Unix (Linux/Max)

Set local shortcuts, replacing *USERNAME*

```bash
touch .bash_profile
echo 'alias nb_bridge='\''echo "ssh -N -L localhost:8800:localhost:8900 USERNAME@bmir-ct-gpu-5.stanford.edu"; ssh -N -L localhost:8800:localhost:8999 USERNAME@bmir-ct-gpu-5.stanford.edu'\''' >> .bash_profile
source .bash_profile
```

Connect to the server  *(requires the Stanford VPN; will ask for your password )*

```bash
ssh USERNAME@bmir-ct-gpu-5.stanford.edu
```

###### Windows

Get PuTTY [https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html).

Connect to the server *(requires the Stanford VPN; will ask for your password )*

```bash
plink.exe -ssh USERNAME@bmir-ct-gpu-5.stanford.edu
```



###### (All) Setup session on the server

Install python

`````````bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
`````````

Download project codebase *(will ask for my password)*

```bash
git clone https://marcthib@bitbucket.org/gevaertlab/romain.git
```

Set up local shortcuts, replacing *USERNAME*

```bash
touch .bash_profile
echo 'alias nb_server="jupyter notebook --no-browser --ip=0.0.0.0 --port=8999"' >> .bash_profile
echo 'export PYTHONPATH="${PYTHONPATH}:/home/USERNAME/romain"' >> .bash_profile
source .bash_profile
```

Install required packages

```bash
conda install -c conda-forge pydicom

while read requirement; do conda install --yes $requirement; done < romain/rtog/radiologist_classification/requirements.txt
```
