import os
import sys
basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basepath)
import numpy as np, os, scipy, scipy.spatial, matplotlib.pyplot as plt, IPython.display as ipd
from itertools import groupby
from numba import jit
import librosa
import libfmp.c3, libfmp.c5
import pandas as pd, pickle, re
from numba import jit
import torch
import torch.utils.data
import torch.nn as nn
from torchinfo import summary
from libdl.data_loaders import dataset_context_segm
from libdl.nn_models import basic_cnn_segm_blank_logsoftmax
from libdl.nn_losses import mctc_we_loss
from libdl.metrics import early_stopping, calculate_eval_measures, calculate_mpe_measures_mireval
import logging


################################################################################
#### Set experimental configuration ############################################
################################################################################

# Get experiment name from script name
curr_filepath = sys.argv[0]
expname = curr_filename = os.path.splitext(os.path.basename(curr_filepath))[0]
print(' ... running experiment ' + expname)

# Which steps to perform
do_train = False
do_val = False
do_test = True
store_results_filewise = True

# Specify model ################################################################

num_octaves_inp = 6
# num_output_bins, min_pitch = 72, 24
num_output_bins, min_pitch = 12, 60
model_params = {'n_chan_input': 6,
                'n_chan_layers': [20,20,10,1],
                'n_ch_out': 2,
                'n_bins_in': num_octaves_inp*12*3,
                'n_bins_out': num_output_bins,
                'a_lrelu': 0.3,
                'p_dropout': 0.2
                }


# Set evaluation measures to compute while testing #############################
if do_test:
    eval_thresh = 0.5
    eval_measures = ['precision', 'recall', 'f_measure', 'cosine_sim', 'binary_crossentropy', \
            'euclidean_distance', 'binary_accuracy', 'soft_accuracy', 'accum_energy', 'roc_auc_measure', 'average_precision_score']


# Specify paths and splits #####################################################
path_data_basedir = os.path.join(basepath, 'data')
path_data = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'hcqt_hs512_o6_h5_s1')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitchclass_hs512')
path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitchclass_hs512_nooverl')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitchclass_hs512_shorten75')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitchclass_hs512_shorten50')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitchclass_hs512_shorten25')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_nooverl')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_shorten75')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_shorten50')
# path_annot = os.path.join(path_data_basedir, 'Schubert_Winterreise', 'pitch_hs512_shorten25')

# Where to save results
dir_output = os.path.join(basepath, 'experiments', 'results_filewise')
fn_output = expname + '.csv'
path_output = os.path.join(dir_output, fn_output)


# Where to save logs
fn_log = expname + '.txt'
path_log = os.path.join(basepath, 'experiments', 'logs', fn_log)

# Log basic configuration
logging.basicConfig(filename=path_log, filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.info('Logging experiment ' + expname)
logging.info('Experiment config: do training = ' + str(do_train))
logging.info('Experiment config: do validation = ' + str(do_val))
logging.info('Experiment config: do testing = ' + str(do_test))
if do_test:
    logging.info('Save filewise results = ' + str(store_results_filewise) + ', in folder ' + path_output)


################################################################################
#### Start experiment ##########################################################
################################################################################


# Generate training dataset ####################################################
if do_val:
    assert do_train, 'Validation without training not possible!'
train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
val_versions = ['FI66', 'TR99']
test_versions = ['HU33', 'SC06']


#### START TESTING #############################################################

if do_test:
    logging.info('\n \n ###################### START TESTING ###################### \n')

    n_files = 0
    total_measures = np.zeros(len(eval_measures))
    total_measures_mireval = np.zeros((14))
    n_kframes = 0 # number of frames / 10^3
    framewise_measures = np.zeros(len(eval_measures))
    framewise_measures_mireval = np.zeros((14))

    fs = 22050
    fmin = librosa.note_to_hz('C1')   # MIDI pitch 24
    bins_per_semitone = 3
    bins_per_octave = 12*bins_per_semitone
    num_octaves=6
    num_harmonics=5
    num_subharmonics=1
    center_bins=True

    df = pd.DataFrame([])

    for fn in os.listdir(path_data):
        if any(test_version in fn for test_version in test_versions):

            pitch_hcqt = np.load(os.path.join(path_data, fn))
            targets = np.load(os.path.join(path_annot, fn)).T
            if num_output_bins!=12:
                targets = targets[:, min_pitch:(min_pitch+num_output_bins)]

            # pitch_cqt = pitch_hcqt[1::3, :, 1]
            # pitch_cqt = libfmp.c3.normalize_feature_sequence(np.abs(pitch_cqt), norm='2', threshold=1e-8)

            chroma_cqt = librosa.feature.chroma_cqt(y=None, sr=fs, C=pitch_hcqt[:, :, 1], hop_length=512, fmin=fmin, n_chroma=12, n_octaves=num_octaves, bins_per_octave=bins_per_octave, cqt_mode='full')
            chroma_norm = libfmp.c3.normalize_feature_sequence(np.abs(chroma_cqt), norm='max', threshold=1e-8)

            pred = chroma_norm.T

            targ = targets[:pred.shape[0], :]

            assert pred.shape==targ.shape, 'Shape mismatch! Target shape: '+str(targ.shape)+', Pred. shape: '+str(pred.shape)

            eval_dict = calculate_eval_measures(targ, pred, measures=eval_measures, threshold=eval_thresh, save_roc_plot=False)
            eval_numbers = np.fromiter(eval_dict.values(), dtype=float)

            metrics_mpe = calculate_mpe_measures_mireval(targ, pred, threshold=eval_thresh, min_pitch=min_pitch)
            mireval_measures = [key for key in metrics_mpe.keys()]
            mireval_numbers = np.fromiter(metrics_mpe.values(), dtype=float)

            n_files += 1
            total_measures += eval_numbers
            total_measures_mireval += mireval_numbers

            kframes = targ.shape[0]/1000
            n_kframes += kframes
            framewise_measures += kframes*eval_numbers
            framewise_measures_mireval += kframes*mireval_numbers

            res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, [fn] + eval_numbers.tolist() + mireval_numbers.tolist()))
            df = df.append(res_dict, ignore_index=True)

            logging.info('file ' + str(fn) + ' tested. Cosine sim: ' + str(eval_dict['cosine_sim']))


    logging.info('### Testing done. Results: ######################################## \n')

    mean_measures = total_measures/n_files
    mean_measures_mireval = total_measures_mireval/n_files
    k_meas = 0
    for meas_name in eval_measures:
        logging.info('Mean ' + meas_name + ':   ' + str(mean_measures[k_meas]))
        k_meas+=1
    k_meas = 0
    for meas_name in mireval_measures:
        logging.info('Mean ' + meas_name + ':   ' + str(mean_measures_mireval[k_meas]))
        k_meas+=1

    res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FILEWISE MEAN'] + mean_measures.tolist() + mean_measures_mireval.tolist()))
    df = df.append(res_dict, ignore_index=True)


    logging.info('\n')

    framewise_means = framewise_measures/n_kframes
    framewise_means_mireval = framewise_measures_mireval/n_kframes
    k_meas = 0
    for meas_name in eval_measures:
        logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means[k_meas]))
        k_meas+=1
    k_meas = 0
    for meas_name in mireval_measures:
        logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means_mireval[k_meas]))
        k_meas+=1

    res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FRAMEWISE MEAN'] + framewise_means.tolist() + framewise_means_mireval.tolist()))
    df = df.append(res_dict, ignore_index=True)

    df.to_csv(path_output)
