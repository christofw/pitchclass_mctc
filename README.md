# multipitch_mctc

This is a pytorch code repository accompanying the following paper:  

> Christof Weiß and Geoffroy Peeters  
> _Training Deep Pitch-Class Representations With a Multi-Label CTC Loss_  
>  Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2021  

This repository only contains exemplary code and pre-trained models for most of the paper's experiments as well as some individual examples. All datasets used in the paper are publicly available (at least partially), e.g. our main datasets:
* [Schubert Winterreise Dataset (SWD)](https://zenodo.org/record/5139893#.YWRcktpBxaQ)
* [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html)
* [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro)  
For details and references, please see the paper.

## Feature extraction and prediction (Jupyter notebooks)

In this top folder, three Jupyter notebooks demonstrate how to 
* preprocess audio files for running our models (_01_precompute_features_),
* load a pretrained model for predicting pitches (_02_predict_with_pretrained_model_),
* generate the visualizations of the paper's Figure 5 (_03_visualize_pitch_class_features_).

## Experiments from the paper (Python scripts)

In the _experiments_ folder, all experimental scripts as well as the log files (subfolder _logs_) and the filewise results (subfolder _results_filewise_) can be found. The folder _models_pretrained_ contains pre-trained models for the main experiments. The subfolder _predictions_ contains exemplary model predictions for two of the experiments. Plese note that re-training requires a GPU as well as the pre-processed training data (see the notebook _01_precompute_features_ for an example). Any script must be started from the repository top folder path in order to get the relative paths working correctly.

The experiment files' names relate to the paper's results in the following way:

### Experiment 1 (Table 3) - Loss and model variants

* _exp118b_traintest_schubert_sctcthreecomp_pitch.py_ (All-Zero baseline)
* _exp136f2_traintest_schubert_librosa_pitchclass_maxnorm.py_ (CQT-Chroma baseline)
* _exp136b_traintest_schubert_sctcthreecomp_pitchclass.py_ (Separable CTC (SCTC) loss)
* _exp136d_traintest_schubert_mctcnethreecomp_pitchclass.py_ (Non-Epsilon MCTC (MCTC:NE) loss)
* _exp136e_traintest_schubert_mctcwe_pitchclass.py_ (MCTC with epsilon (MCTC:WE) loss)
* _exp136h_traintest_schubert_aligned_pitchclass.py_ (Strongly-aligned training (BCE loss))

### Experiment 2 (Figure 4) - Cross-dataset experiment

* _exp131b_trainmaestromunet_testmix_mctcwe_pitchclass_basiccnn_normtargl_SGD.py_ (Train MusicNet & MAESTRO, test others, MCTC)
* _exp131e_trainmaestromunet_testmix_aligned_pitchclass_basiccnn_SGD.py_ (Train MusicNet & MAESTRO, test others, aligned)
* _exp137a_trainmix_testmusicnet_mctcwe_pitchclass_basiccnn.py_ (Test MusicNet, train others, MCTC)
* _exp137b_trainmix_testmusicnet_aligned_pitchclass_basiccnn_segmmodel.py_ (Test MusicNet, train others aligned)
* _exp138a_trainmix_testmaestro_mctcwe_pitchclass_basiccnn.py_ (Test MAESTRO, train others MCTC)
* _exp138b_trainmix_testmaestro_aligned_pitchclass_basiccnn_segmmodel.py_ (Test MAESTRO, train others aligned)

### Application: Visualization (Figure 5)

* Please see the Jupyter Notebook _03_visualize_pitch_class_features_.
