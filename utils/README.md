# Preprocessing MRI Data for Deep Learning

## Tools

### FSL

#### Installation

First, download the FSL install script.

```bash 
wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py
```
Next, make sure `/home/<YOUR USERNAME>/pkg` exists and then install FSL. 

```bash
python2 fslinstaller.py
```

When asked to choose the FSL install location, choose `/home/<YOUR USERNAME>/pkg/fsl`. Finally, add the following lines to your `~/.bashrc` file:

```bash
export PATH="$PATH":"$HOME"/pkg/fsl/bin
export FSLDIR="$HOME"/pkg/fsl
. "$FSLDIR"/etc/fslconf/fsl.sh
```

#### Usage

Here is an overview of some of the most useful tools that FSL provides.

##### bet

`bet` is a CLI program for brain extraction, AKA skull stripping. Online documentation can be found [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide). The most important options are summarized below.

The syntax for `bet` is

```bash
bet <input> <output> [options]
```

The input needs to be a `.nii` or a `.nii.gz` file. The output is a `.nii.gz` file. The most important option is `-f <f>`. `f` can range from 0 to 1, and its default value is 0.5. Smaller values of `f` give larger brain outlines.

`-S` reduces the number of eye voxels. `-B` reduces the number of neck voxels. `-Z` makes `bet` work better with a small number of axial slices. More information on these and other options can be found in the online documentation.

##### fsl\_anat

`fsl_anat` is a general purpose tool for analyzing structural MRI images. Online documentation can be found [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/fsl_anat). We use it for reorientation standard orientation, FOV reduction, and bias-field correction. `fsl_anat` can also do registration and segmentation, but that is usually excessive for our purposes.

We run `fsl_anat` as follows:

```bash
fsl_anat -i <input> -o <output> -t <modality> --noreg --nononlinreg --noseg --nosubcortseg
```

The input needs to be a `.nii` or a `.nii.gz` file. `fsl_anat` creates a directory `<output>.anat`. The modality can be either `T1` or `T2`. Be advised that this command takes about 5 minutes to run!

##### slices

`slices` is an easy-to-use viz tool.

We run `slices` as follows:

```bash
slices <input> -o <output>
```

The input needs to be a `.nii` or a `.nii.gz` file. The output is a `.gif` image.

### NiftyReg

#### Installation

NiftyReg is already installed on the GPU cluster. Rejoice!

#### Usage

##### reg\_aladin

`reg_aladin` is a general purpose registration tool. Online documentation can be found [here](http://cmictig.cs.ucl.ac.uk/wiki/index.php/Reg_aladin). For deep learning, we only require brain-by-brain registration, not global registration. `reg_aladin` provides the ability to only perform a rigid registration.

We run `reg_aladin` as follows:

```bash
reg_aladin -ref <reference> -flo <input> -aff <xform> -res <output> -rigOnly
```

The input and reference need to be `.nii` or `.nii.gz` files. The resulting xform is a `.txt` file and the output is a `.nii` file. The `-rigOnly` option specifies that a rigid registration will be performed.

## Code

The file `process_rtog_nii.py` provides wrapper Python functions for `fsl_anat`, `bet`,  and `reg_aladin` programs and a high-level preprocessing script for RTOG data that calls all three programs. Information can be found in the docstrings within.

The files `../rtog/preprocess_images.ipynb` and `../rtog/preprocess_images_script.py` contain higher-level pipelines for processing `.nii` images found in the `/local-scratch/"$USER"_scratch/rtog` directory on the GPU server.

Currently, images have been preprocessed with default `f` and `g` values and without higher-level `bet` options. Some of the images caused errors because of missing modalities or other issues. The corresponding patient numbers are listed in the files mentioned in the previous paragraph.

Contact Raghav Subramaniam (raghavs511@gmail.com) if you have questions about this code.
