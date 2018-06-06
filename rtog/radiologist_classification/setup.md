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

- Connect to the Jupyter server on your internet browser (Safari, Chrome, Firefox), by following the link [localhost:8701/](localhost:8701/).

  Asked for a password, enter: **radiology**

- Navigate to `rtog/radiologist_classification/mri_viewer-rtog.ipynb`



- First, hit **`RUN`** for the first 3 cells. 
- Next, either :
  - enter a `DESCRIPTION_NUMBER` and hit **`RUN`** on the 2 cells. It will show you the 3D MRI scan. It will also print the Description of the file. *This will only work for the T1preVpost spreadsheet.*
  - or enter a `FOLDER_NAME` and hit **`RUN`** on the 2 cells. It will show you the 3D MRI scan. It will also print the Description of the file. *This will work for all spreadsheets*.



- Debug: 
  - if an interactive viewer freezes, you can re-**`RUN`** the cell. 
  - if the page gets buggy, go to the toolbar on top of the page, and hit `Kernel/Restart`.
  - if you can not connect to the page, on anything else happens, e-mail me and I'll set it back up as soon as possible! 