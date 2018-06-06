# Setup for visualizing RTOG images from the Gevaert lab



##### Unix (Linux/Max)
Connect to the server and run a distant Jupyter instance *(requires the Stanford VPN; will ask for your password )*

```bash
ssh -N -L localhost:8701:localhost:8700 USERNAME@bmir-ct-gpu-5.stanford.edu
```


##### Windows

Connect to the server and run a distant Jupyter instance *(requires the Stanford VPN; will ask for your password )*. You will need PuTTY ([https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)).

```
plink.exe -N -L localhost:8701:localhost:8700 USERNAME@bmir-ct-gpu-5.stanford.edu
```



##### (All) Launch the visualization UI 

Connect to the Jupyter server on your internet browser (Safari, Chrome, Firefox), by following the link [localhost:8701/](localhost:8701/).

Asked for a password, enter: **radiology**



Navigate to `rtog/radiologist_classification/mri_viewer-rtog.ipynb`


