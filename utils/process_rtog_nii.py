"""
A file containing methods to process RTOG .nii files for use in deep learning.
"""

import os
import shutil
import subprocess
import tempfile

def _preproc(nii, modality, nii_out):
    """
    Run fsl_anat to preprocess a brain.

    Args:
    nii: path to a .nii file
    modality: modality of the brain ('T1' or 'T2')
    nii_out: path to output processed .nii file to
    """
    tmp_dir = os.path.abspath('tmp')
    os.mkdir(tmp_dir)
    subprocess.call(['fsl_anat',
                     '-i', nii,
                     '-o', os.path.join(tmp_dir, modality),
                     '-t', modality,
                     '--noreg', '--nononlinreg',
                     '--noseg', '--nosubcortseg'])
    os.rename(os.path.join(tmp_dir, 
              '{}.anat'.format(modality),
              '{}_biascorr.nii.gz'.format(modality)),
              nii_out + '.gz')
    shutil.rmtree(tmp_dir)

def _skull_strip(nii, nii_out, f=0.5, g=0.):
    """
    Run bet to skull strip a brain.

    Args:
    nii: path to a .nii file
    nii_out: path to output processed .nii file to
    f: the f param for bet (default: 0.5)
    g: the g param for bet (default: 0.)
    """
    subprocess.call(['bet', nii, nii_out, '-f', str(f), '-g', str(g)])

def _register(nii, ref, nii_out):
    """
    Run reg_aladin to rigidly register a brain to a reference.

    Args:
    nii: path to a .nii file
    ref: path to a reference .nii file
    nii_out: path to output processed .nii file to
    """
    aff_file = os.path.abspath('reg.txt')
    subprocess.call(['reg_aladin',
                     '-ref', ref + '.gz',
                     '-flo', nii + '.gz',
                     '-aff', aff_file,
                     '-res', nii_out,
                     '-rigOnly'])
    os.remove(aff_file)

################################################################################

def process_rtog_nii(nii_dir, f=0.5, g=0.):
    """
    Process a pair of t1c and flair RTOG .nii files.

    Args:
    nii_dir: a directory containing two files, t1c.nii and flair.nii.
    f: the f param for bet (default: 0.5)
    g: the g param for bet (default: 0.)
    
    This method creates two files t1c_proc.nii and flair_proc.nii in nii_dir.
    """
    nii_dir = os.path.abspath(nii_dir)
    _preproc(os.path.join(nii_dir, 't1c.nii'), 'T1', 
             os.path.join(nii_dir, 't1c_fsl_anat.nii'))
    _preproc(os.path.join(nii_dir, 'flair.nii'), 'T2', 
             os.path.join(nii_dir, 'flair_fsl_anat.nii'))
    _skull_strip(os.path.join(nii_dir, 't1c_fsl_anat.nii'),
                 os.path.join(nii_dir, 't1c_bet.nii'), f=f, g=g)
    _skull_strip(os.path.join(nii_dir, 'flair_fsl_anat.nii'),
                 os.path.join(nii_dir, 'flair_bet.nii'), f=f, g=g)
    _register(os.path.join(nii_dir, 'flair_bet.nii'),
              os.path.join(nii_dir, 't1c_bet.nii'),
              os.path.join(nii_dir, 'flair_reg_aladin.nii'))
    subprocess.call(['gunzip', os.path.join(nii_dir, 't1c_bet.nii.gz')])
    os.rename(os.path.join(nii_dir, 't1c_bet.nii'),
              os.path.join(nii_dir, 't1c_proc.nii'))
    os.rename(os.path.join(nii_dir, 'flair_reg_aladin.nii'),
              os.path.join(nii_dir, 'flair_proc.nii'))
    os.remove(os.path.join(nii_dir, 't1c_fsl_anat.nii.gz'))
    os.remove(os.path.join(nii_dir, 'flair_fsl_anat.nii.gz'))
    os.remove(os.path.join(nii_dir, 'flair_bet.nii.gz'))
