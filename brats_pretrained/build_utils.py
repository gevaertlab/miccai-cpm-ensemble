from __future__ import print_function

import os
import random
import shutil

def build_config_file(name, _dict):
    """ Build a config file for an experiment.

    params:
    name: the name of the experiment
    _dict: the dictionary of experiment parameters
    """
    filename = '../../config/' + name + '.cfg'

    for key in _dict:
        line = str(key) + ' ' + str(_dict[key])
        with open(filename, 'ab') as f:
            print(line, file=f)

def build_sbatch_file(name, time, mem, script):
    """ Build an sbatch file for an experiment.

    params:
    name: the name of the experiment
    time: how long (in hours) the experiment will take
    mem: how much (in GB) memory the experiment will require
    script: the name of the script to run for the experiment
    """

    filename = '../../../' + name + '.sbatch'

    with open(filename, 'wb') as f:
        print('#!/bin/bash', file=f)
        print(file=f)
        print('#SBATCH --job-name=' + name, file=f)
        print('#SBATCH --output=' + name + '.out', file=f)
        print('#SBATCH --error=' + name + '.err', file=f)
        print(file=f)
        print('#SBATCH --time=' + time_str(time), file=f)
        print('#SBATCH --mem=' + mem_str(mem), file=f)
        print(file=f)
        print('#SBATCH --qos=gpu', file=f)
        print('#SBATCH --partition=gpu', file=f)
        print('#SBATCH --gres=gpu:1', file=f)
        print(file=f)
        print('module load tensorflow/0.9.0', file=f)
        print('cd tumor-seg', file=f)
        print(file=f)
        print('python {} --cfg-path=config/{}.cfg'
              .format(script, name), file=f)

def time_str(time):
    """ Convert a time to a string.

    params:
    time: a time in hours
    """

    return str(time) + ':00:00'

def mem_str(mem):
    """ Convert a mem to a string.

    params:
    mem: a mem in GB
    """

    return str(mem) + '000'

def build_prime_dataset(fine_tune_path, base_path, frac):
    """ 
    Put together a dataset for a priming experiment.

    params:
    fine_tune_path: the path to the original testing set
    base_path: the base path for the priming set and the new testing set.
    frac: the fraction of original testing examples to use for priming.
    """
    
    train_path = base_path + '/train'
    test_path = base_path + '/test'

    shutil.copytree(fine_tune_path, test_path)

    test_ex = os.listdir(test_path)
    train_ex = random.sample(test_ex, int(frac * len(test_ex)))

    for ex in train_ex:
        ex_test_path = os.path.join(test_path, ex)
        ex_train_path = os.path.join(train_path, ex)
        shutil.move(ex_test_path, ex_train_path)
