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
from libdl.data_loaders import dataset_context
from libdl.nn_models import deep_cnn_segm_sigmoid
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
do_train = True
do_val = True
do_test = True
store_results_filewise = True
store_predictions = True

# Set training parameters
train_dataset_params = {'context': 75,
                        'stride': 100,
                        'compression': 10
                        }
val_dataset_params = {'context': 75,
                      'stride': 100,
                      'compression': 10
                      }
test_dataset_params = {'context': 75,
                       'stride': 1,
                       'compression': 10
                      }
train_params = {'batch_size': 25,
                'shuffle': True,
                'num_workers': 16
                }
val_params = {'batch_size': 50,
              'shuffle': False,
              'num_workers': 16
              }
test_params = {'batch_size': 50,
              'shuffle': False,
              'num_workers': 8
              }


# Specify model ################################################################

num_octaves_inp = 6
# num_output_bins, min_pitch = 72, 24
num_output_bins, min_pitch = 12, 60
model_params = {'n_chan_input': 6,
                'n_chan_layers': [20,20,10,1],
                'n_prefilt_layers': 5,
                'residual': True,
                'n_bins_in': num_octaves_inp*12*3,
                'n_bins_out': num_output_bins,
                'a_lrelu': 0.3,
                'p_dropout': 0.2
                }

if do_train:

    max_epochs = 50

# Specify criterion (loss) #####################################################
    criterion = torch.nn.BCELoss(reduction='mean')
    # criterion = sctc_loss_threecomp()
    # criterion = sctc_loss_twocomp()
    # criterion = mctc_ne_loss_twocomp()
    # criterion = mctc_ne_loss_threecomp()
    # criterion = mctc_we_loss()



# Set optimizer and parameters #################################################
    optimizer_params = {'name': 'SGD',
                        'initial_lr': 0.01,
                        'momentum': 0.9}
    # optimizer_params = {'name': 'Adam',
    #                     'initial_lr': 0.01,
    #                     'betas': [0.9, 0.999]}
    # optimizer_params = {'name': 'AdamW',
    #                     'initial_lr': 0.01,
    #                     'betas': (0.9, 0.999),
    #                     'eps': 1e-08,
    #                     'weight_decay': 0.01,
    #                     'amsgrad': False}


# Set scheduler and parameters #################################################
    # scheduler_params = {'use_scheduler': True,
    #                     'name': 'LambdaLR',
    #                     'start_lr': 1,
    #                     'end_lr': 1e-2,
    #                     'n_decay': 20,
    #                     'exp_decay': .5
    #                     }
    scheduler_params = {'use_scheduler': True,
                        'name': 'ReduceLROnPlateau',
                        'mode': 'min',
                        'factor': 0.5,
                        'patience': 4,
                        'threshold': 0.0001,
                        'threshold_mode': 'rel',
                        'cooldown': 0,
                        'min_lr': 1e-6,
                        'eps': 1e-08,
                        'verbose': False
                        }


# Set early_stopping and parameters ############################################
    early_stopping_params = {'use_early_stopping': True,
                             'mode': 'min',
                             'min_delta': 1e-5,
                             'patience': 12,
                             'percentage': False
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

# Where to save predictions
dir_predictions = os.path.join(basepath, 'predictions', expname)

# Where to save logs
fn_log = expname + '.txt'
path_log = os.path.join(basepath, 'experiments', 'logs', fn_log)

# Log basic configuration
logging.basicConfig(filename=path_log, filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.info('Logging experiment ' + expname)
logging.info('Experiment config: do training = ' + str(do_train))
logging.info('Experiment config: do validation = ' + str(do_val))
logging.info('Experiment config: do testing = ' + str(do_test))
logging.info("Training set parameters: {0}".format(train_dataset_params))
logging.info("Validation set parameters: {0}".format(val_dataset_params))
logging.info("Test set parameters: {0}".format(test_dataset_params))
if do_train:
# Log criterion, optimizer, and scheduler ######################################
    logging.info(' --- Training config: ----------------------------------------- ')
    logging.info('Maximum number of epochs: ' + str(max_epochs))
    logging.info('Criterion (Loss): ' + criterion.__class__.__name__)
    logging.info("Optimizer parameters: {0}".format(optimizer_params))
    logging.info("Scheduler parameters: {0}".format(scheduler_params))
    logging.info("Early stopping parameters: {0}".format(early_stopping_params))
if do_test:
    logging.info("Test parameters: {0}".format(test_params))
    logging.info('Save filewise results = ' + str(store_results_filewise) + ', in folder ' + path_output)
    logging.info('Save model predictions = ' + str(store_predictions) + ', in folder ' + dir_predictions)


################################################################################
#### Start experiment ##########################################################
################################################################################

# CUDA for PyTorch #############################################################
use_cuda = torch.cuda.is_available()
assert use_cuda, 'No GPU found! Exiting.'
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
logging.info('CUDA use_cuda: ' + str(use_cuda))
logging.info('CUDA device: ' + str(device))

# Where to save models
dir_models = os.path.join(basepath, 'models')
fn_model = expname + '.pt'
path_trained_model = os.path.join(dir_models, fn_model)

logging.info("Training parameters: {0}".format(train_params))
logging.info('Trained model saved in ' + path_trained_model)

# Specify and log model config #################################################
mp = model_params
model = deep_cnn_segm_sigmoid(n_chan_input=mp['n_chan_input'], n_chan_layers=mp['n_chan_layers'], n_prefilt_layers=mp['n_prefilt_layers'], \
    residual=mp['residual'], n_bins_in=mp['n_bins_in'], n_bins_out=mp['n_bins_out'], a_lrelu=mp['a_lrelu'], p_dropout=mp['p_dropout'])
model.to(device)

logging.info(' --- Model config: -------------------------------------------- ')
logging.info('Model: ' + model.__class__.__name__)
logging.info("Model parameters: {0}".format(model_params))
logging.info('\n' + str(summary(model, input_size=(1, 6, 174, 216))))


# Generate training dataset ####################################################
if do_val:
    assert do_train, 'Validation without training not possible!'
train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
val_versions = ['FI66', 'TR99']
test_versions = ['HU33', 'SC06']

all_train_fn = []
all_train_sets = []
all_val_fn = []
all_val_sets = []

if do_train:
    for fn in os.listdir(path_data):
        if any(train_version in fn for train_version in train_versions):
            all_train_fn.append(fn)
            inputs = torch.from_numpy(np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0)))
            targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)).T)
            if num_output_bins!=12:
                targets = targets[:, min_pitch:(min_pitch+num_output_bins)]
            curr_dataset = dataset_context(inputs, targets, train_dataset_params)
            all_train_sets.append(curr_dataset)
            logging.info(' - file ' + str(fn) + ' added to training set.')
        if do_val:
            if any(val_version in fn for val_version in val_versions):
                all_val_fn.append(fn)
                inputs = torch.from_numpy(np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0)))
                targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)).T)
                if num_output_bins!=12:
                    targets = targets[:, min_pitch:(min_pitch+num_output_bins)]
                curr_dataset = dataset_context(inputs, targets, val_dataset_params)
                all_val_sets.append(curr_dataset)
                logging.info(' - file ' + str(fn) + ' added to validation set.')

    train_set = torch.utils.data.ConcatDataset(all_train_sets)
    train_loader = torch.utils.data.DataLoader(train_set, **train_params)
    logging.info('Training set & loader generated, length ' + str(len(train_set)))

    if do_val:
        val_set = torch.utils.data.ConcatDataset(all_val_sets)
        val_loader = torch.utils.data.DataLoader(val_set, **val_params)
        logging.info('Validation set & loader generated, length ' + str(len(val_set)))

# Set training configuration ###################################################

    criterion.to(device)

    op = optimizer_params
    if op['name']=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=op['initial_lr'], momentum=op['momentum'])
    elif op['name']=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=op['initial_lr'], betas=op['betas'])
    elif op['name']=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=op['initial_lr'], betas=op['betas'], eps=op['eps'], weight_decay=op['weight_decay'], amsgrad=op['amsgrad'])

    sp = scheduler_params
    if sp['use_scheduler'] and sp['name']=='LambdaLR':
        start_lr, end_lr, n_decay, exp_decay = sp['start_lr'], sp['end_lr'], sp['n_decay'], sp['exp_decay']
        polynomial_decay = lambda epoch: ((start_lr - end_lr) * (1 - min(epoch, n_decay)/n_decay) ** exp_decay ) + end_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)
    elif sp['use_scheduler'] and sp['name']=='ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sp['mode'], \
        factor=sp['factor'], patience=sp['patience'], threshold=sp['threshold'], threshold_mode=sp['threshold_mode'], \
        cooldown=sp['cooldown'], eps=sp['eps'], min_lr=sp['min_lr'], verbose=sp['verbose'])

    ep = early_stopping_params
    if ep['use_early_stopping']:
        es = early_stopping(mode=ep['mode'], min_delta=ep['min_delta'], patience=ep['patience'], percentage=ep['percentage'])


#### START TRAINING ############################################################

    logging.info('\n \n ###################### START TRAINING ###################### \n')

    # Loop over epochs
    for epoch in range(max_epochs):
        accum_loss, n_batches = 0, 0
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            y_pred = model(local_batch)
            loss = criterion(y_pred, local_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()
            n_batches += 1

        train_loss = accum_loss/n_batches

        if do_val:
            accum_val_loss, n_val = 0, 0
            for local_batch, local_labels in val_loader:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                y_pred = model(local_batch)
                loss = criterion(y_pred, local_labels)

                accum_val_loss += loss.item()
                n_val += 1
            val_loss = accum_val_loss/n_val

        # Log epoch results
        if sp['use_scheduler'] and sp['name']=='LambdaLR' and do_val:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + \
            ', Val Loss: ' + "{:.4f}".format(val_loss) + ' with lr: ' + "{:.5f}".format(scheduler.get_last_lr()[0]))
            scheduler.step()
        elif sp['use_scheduler'] and sp['name']=='ReduceLROnPlateau' and do_val:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + \
            ', Val Loss: ' + "{:.4f}".format(val_loss) + ' with lr: ' + "{:.5f}".format(optimizer.param_groups[0]['lr']))
            scheduler.step(val_loss)
        elif sp['use_scheduler'] and sp['name']=='LambdaLR':
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + ', with lr: ' + "{:.5f}".format(scheduler.get_last_lr()[0]))
            scheduler.step()
        elif sp['use_scheduler'] and sp['name']=='ReduceLROnPlateau':
            assert False, 'Scheduler ' + sp['name'] + ' requires validation set!'
        else:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + ', with lr: ' + "{:.5f}".format(optimizer_params['initial_lr']))

        # Perform early stopping
        if ep['use_early_stopping'] and epoch==0:
            torch.save(model.state_dict(), path_trained_model)
            logging.info('  .... model of epoch 0 saved.')
        elif ep['use_early_stopping'] and epoch>0:
            if es.curr_is_better(val_loss):
                torch.save(model.state_dict(), path_trained_model)
                logging.info('  .... model of epoch #' + str(epoch) + ' saved.')
        if ep['use_early_stopping'] and es.step(val_loss):
            break

    if not ep['use_early_stopping']:
        torch.save(model.state_dict(), path_trained_model)

    logging.info(' ### trained model saved in ' + path_trained_model + ' \n')


#### START TESTING #############################################################

if do_test:
    logging.info('\n \n ###################### START TESTING ###################### \n')

    # Load pretrained model
    if (not do_train) or (do_train and ep['use_early_stopping']):
        model.load_state_dict(torch.load(path_trained_model))
    if not do_train:
        logging.info(' ### trained model loaded from ' + path_trained_model + ' \n')
    model.eval()

    # Set test parameters
    half_context = test_dataset_params['context']//2

    n_files = 0
    total_measures = np.zeros(len(eval_measures))
    total_measures_mireval = np.zeros((14))
    n_kframes = 0 # number of frames / 10^3
    framewise_measures = np.zeros(len(eval_measures))
    framewise_measures_mireval = np.zeros((14))

    df = pd.DataFrame([])

    for fn in os.listdir(path_data):
        if any(test_version in fn for test_version in test_versions):

            inputs = np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0))
            targets = np.load(os.path.join(path_annot, fn)).T
            inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context+1), (0, 0))))
            targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context+1), (0, 0))))

            test_set = dataset_context(inputs_context, targets_context, test_dataset_params)
            test_generator = torch.utils.data.DataLoader(test_set, **test_params)

            pred_tot = np.zeros((0, num_output_bins))
            for test_batch, test_labels in test_generator:
                # Transfer to GPU
                test_batch = test_batch.to(device)

                # Model computations
                y_pred = model(test_batch)
                y_pred = y_pred.to('cpu')
                # pred = torch.squeeze(y_pred).detach().numpy()
                pred = torch.squeeze(torch.squeeze(y_pred,2),1).detach().numpy()
                pred_tot = np.append(pred_tot, pred, axis=0)

            pred = pred_tot
            targ = targets

            assert pred.shape==targ.shape, 'Shape mismatch! Target shape: '+str(targ.shape)+', Pred. shape: '+str(pred.shape)

            if not os.path.exists(os.path.join(dir_predictions)):
                os.makedirs(os.path.join(dir_predictions))
            np.save(os.path.join(dir_predictions, fn[:-4]+'.npy'), pred)

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
