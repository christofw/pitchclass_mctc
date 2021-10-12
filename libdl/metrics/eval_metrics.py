import numpy as np, os, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa
import libfmp.c3, libfmp.c5
import libdl.data_preprocessing
from sklearn import metrics as sk_metrics
from mir_eval import multipitch as mpe

def calculate_single_measure(targets, predictions, measure, threshold=0.5, save_roc_plot=False, path_output='roc.pdf'):
    """ Calculates evaluation measures for a target and a prediction array,
    with frames (times) as the first and bins (pitch, pitch class, ...) as
    the second dimension

    Args:
        targets:           Binary array of targets, size (n_frames, n_bins)
        predictions:       Prediction array (probabilities between zero and one of
                           same size
        measure:           Type of evaluation measure. Possible types
                            - precision
                            - recall
                            - f_measure
                            - cosine_sim
                            - binary_crossentropy
                            - euclidean_distance
                            - binary_accuracy
                            - soft_accuracy
                            - accum_energy
                            - roc_auc_measure
                            - average_precision_score (=improved AUC measure)
        threshold:          Threshold value for measures that require binarization
                            (precision, recall, f_meas, binary_accuracy), default 0.5
        save_roc_plot:      Bool for saving a pdf of the ROC curve
        path_output:        path to save ROC curve plot

    Returns:
        measure_val:        Value of the evaluation measure (averaged over frames)
    """

    targ = targets
    pred = predictions

    assert targ.shape==pred.shape, 'Error: Targets and predictions have different shape!'

    if np.mod(targ.shape[1], 12)!=0:
        print('WARNING: Shape of input is ' + str(targ.shape) + \
        ', expect features (bins) as second dimension. Please make sure that size is correct!')

    thresh = threshold
    eps = np.finfo(float).eps
    threshold_L2norm = 1e-10
    pred_thresh = pred >= thresh

    if measure=='precision':
        precision, recall, f_measure, TP, FP, FN = libfmp.c5.compute_eval_measures(targ, pred_thresh)
        measure_val = precision

    elif measure=='recall':
        precision, recall, f_measure, TP, FP, FN = libfmp.c5.compute_eval_measures(targ, pred_thresh)
        measure_val = recall

    elif measure=='f_measure':
        precision, recall, f_measure, TP, FP, FN = libfmp.c5.compute_eval_measures(targ, pred_thresh)
        measure_val = f_measure

    elif measure=='cosine_sim':
        targ_L2 = libfmp.c3.normalize_feature_sequence(targ.T, norm='2', threshold=threshold_L2norm)
        pred_L2 = libfmp.c3.normalize_feature_sequence(pred.T, norm='2', threshold=threshold_L2norm)
        cosine_sim = np.sum(np.multiply(targ_L2, pred_L2))/targ_L2.shape[1]
        measure_val = cosine_sim

    elif measure=='binary_crossentropy':
        binary_crossentropy = -np.mean(targ*np.log2(pred+eps)+(1-targ)*np.log2(1-pred+eps))
        measure_val = binary_crossentropy

    elif measure=='euclidean_distance':
        euclidean_distance = np.mean(np.sqrt(np.sum((targ-pred)**2, axis=1)))
        measure_val = euclidean_distance

    elif measure=='binary_accuracy':
        binary_accuracy = np.mean(pred_thresh==targ)
        measure_val = binary_accuracy

    elif measure=='soft_accuracy':
        soft_accuracy = np.mean(targ*pred + (1-targ)*(1-pred))
        measure_val = soft_accuracy

    elif measure=='accum_energy':
        accum_energy = np.mean(np.sum(targ*pred, axis=1)/(np.sum(targ, axis=1)+eps))
        measure_val = accum_energy

    elif measure=='roc_auc_measure':

        roc_auc_measure = sk_metrics.roc_auc_score(targ.flatten(), pred.flatten())
        measure_val = roc_auc_measure

        if save_roc_plot:
            fpr, tpr, thresholds = sk_metrics.roc_curve(targ.flatten(), pred.flatten(), pos_label=1)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc_measure)
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(path_output)

    elif measure=='average_precision_score':

        average_precision_score = sk_metrics.average_precision_score(targ.flatten(), pred.flatten())
        measure_val = average_precision_score

    else:
        assert False, 'ERROR: Evaluation measure ' + str(measure) + ' not implemented!'

    return measure_val



def calculate_eval_measures(targets, predictions, measures, threshold=0.5, save_roc_plot=False, path_output='roc.pdf'):
    """ Calculates all evaluation measures for a target and a prediction array

    Args:
        targets:           Binary array of targets, size (n_frames, n_bins)
        predictions:       Prediction array (probabilities between zero and one of
                           same size
        measures:          List of evaluation measure types. Possible types
                            - precision
                            - recall
                            - f_measure
                            - cosine_sim
                            - binary_crossentropy
                            - euclidean_distance
                            - binary_accuracy
                            - soft_accuracy
                            - accum_energy
                            - roc_auc_measure
                            - average_precision_score (=improved AUC measure)
        threshold:          Threshold value for measures that require binarization
                            (precision, recall, f_meas, binary_accuracy), default 0.5
        save_roc_plot:      Bool for saving a pdf of the ROC curve
        path_output:        path to save ROC curve plot

    Returns:
        measures_dict:      Dictionary containing names and values of the evaluation
                            measures (averaged over frames)
    """

    measures_dict = {}

    for measure in measures:
        measure_val = calculate_single_measure(targets, predictions, measure, threshold, save_roc_plot, path_output)
        measures_dict[measure] = measure_val

    return measures_dict


def calculate_mpe_measures_mireval(targets, predictions, threshold=0.5, min_pitch=24):
    """ Calculates evaluation measures for multi-pitch estimation using mir_eval

    Args:
        targets:           Binary array of targets, size (n_frames, n_bins)
        predictions:       Prediction array (probabilities between zero and one of
                           same size
        threshold:         Threshold value for measures that require binarization
                           (precision, recall, f_meas, binary_accuracy), default 0.5
        min_pitch:         MIDI pitch corresponding to lowest row of targets / predictions

    Returns:
        measures_dict:      Dictionary containing names and values of the evaluation
                            measures (averaged over frames)
    """

    fs_hcqt = 43.066406250   # TODO: Make this a variable later...

    targ = targets
    pred = predictions
    thresh = threshold
    pred_thresh = pred >= thresh

    ref_time = np.arange(targ.shape[0])/fs_hcqt
    est_time = np.arange(pred_thresh.shape[0])/fs_hcqt

    ref_freqs = [librosa.midi_to_hz(min_pitch + np.squeeze(np.nonzero(targ[k,:]), 0)) for k in range(targ.shape[0])]
    est_freqs = [librosa.midi_to_hz(min_pitch + np.squeeze(np.nonzero(pred_thresh[k,:]), 0)) for k in range(pred_thresh.shape[0])]

    measures_dict = mpe.evaluate(ref_time, ref_freqs, est_time, est_freqs)

    return measures_dict
