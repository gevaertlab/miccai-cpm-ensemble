"""
A file containing methods to process RTOG .nii files for use in deep learning.
"""

import os
import shutil
import subprocess

DEVNULL = open(os.devnull, 'w')

def _preproc(nii, modality, nii_out, working_dir):
    """
    Run fsl_anat to preprocess a brain.

    Args:
    nii: path to a .nii file
    modality: modality of the brain ('T1' or 'T2')
    nii_out: path to output processed .nii file to
    """
    tmp_dir = os.path.join(working_dir, 'tmp')
    try:
        os.mkdir(tmp_dir)
    except FileExistsError:
        pass

    command_list = ['fsl_anat',
                     '-i', nii,
                     '-o', os.path.join(tmp_dir, modality),
                     '-t', modality,
                     '--noreg', '--nononlinreg',
                     '--noseg', '--nosubcortseg']
    print(" ".join(command_list))
    subprocess.call(command_list, stdout=DEVNULL)

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
    command_list = ['bet',
                    nii,
                    nii_out,
                    '-f', str(f),
                    '-g', str(g)]
    print(" ".join(command_list))
    subprocess.call(command_list, stdout=DEVNULL)


def _register(nii, ref, nii_out):
    """
    Run reg_aladin to rigidly register a brain to a reference.

    Args:
    nii: path to a .nii file
    ref: path to a reference .nii file
    nii_out: path to output processed .nii file to
    """
    aff_file = os.path.abspath('reg.txt')
    command_list = ['reg_aladin',
                     '-ref', ref + '.gz',
                     '-flo', nii + '.gz',
                     '-aff', aff_file,
                     '-res', nii_out,
                     '-rigOnly']
    print(" ".join(command_list))
    subprocess.call(command_list, stdout=DEVNULL)
    os.remove(aff_file)


################################################################################

def process_rtog_nii(nii_dir,
                     input_t1c_filename='t1c.nii', output_t1c_filename='t1c_proc.nii',
                     input_flair_filename='flair.nii', output_flair_filename='flair_proc.nii',
                     f=0.5, g=0.):
    """
    Process a pair of t1c and flair RTOG .nii files.

    Args:
    nii_dir: a directory containing two files, t1c.nii and flair.nii.
    f: the f param for bet (default: 0.5)
    g: the g param for bet (default: 0.)
    
    This method creates two files t1c_proc.nii and flair_proc.nii in nii_dir.
    """
    nii_dir = os.path.abspath(nii_dir)

    _preproc(os.path.join(nii_dir, input_t1c_filename),
             'T1',
             os.path.join(nii_dir, 't1c_fsl_anat.nii'),
             nii_dir)
    _preproc(os.path.join(nii_dir, input_flair_filename),
             'T2',
             os.path.join(nii_dir, 'flair_fsl_anat.nii'),
             nii_dir)

    _skull_strip(os.path.join(nii_dir, 't1c_fsl_anat.nii'),
                 os.path.join(nii_dir, 't1c_bet.nii'), f=f, g=g)
    _skull_strip(os.path.join(nii_dir, 'flair_fsl_anat.nii'),
                 os.path.join(nii_dir, 'flair_bet.nii'), f=f, g=g)

    _register(os.path.join(nii_dir, 'flair_bet.nii'),
              os.path.join(nii_dir, 't1c_bet.nii'),
              os.path.join(nii_dir, 'flair_reg_aladin.nii'))

    subprocess.call(['gunzip', os.path.join(nii_dir, 't1c_bet.nii.gz')])

    os.rename(os.path.join(nii_dir, 't1c_bet.nii'),
              os.path.join(nii_dir, output_t1c_filename))
    os.rename(os.path.join(nii_dir, 'flair_reg_aladin.nii'),
              os.path.join(nii_dir, output_flair_filename))

    os.remove(os.path.join(nii_dir, 't1c_fsl_anat.nii.gz'))
    os.remove(os.path.join(nii_dir, 'flair_fsl_anat.nii.gz'))
    os.remove(os.path.join(nii_dir, 'flair_bet.nii.gz'))
