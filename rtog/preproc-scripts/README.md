# RTOG preprocessing scripts

## Scripts

`update-header-dz` is a bash script that updates the axial voxel size in a NifTI file header. We need this because the RTOG dataset has incorrect header information.

`test-f-g` is a Python script that sweeps over various `f` and `g` parameters for bet and runs skull stripping code. This involves calling `update-header-dz`, running basic preprocessing, and running vanilla `bet`.

`process-brain` is a bash script that runs skull stripping and registration for a single brain.

`process-dataset` runs `process-brain` for every brain in the dataset.
