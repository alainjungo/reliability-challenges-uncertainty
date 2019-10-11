import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
####################################
# dirs and path required to be set #
####################################
BRATS_ORIG_DATA_DIR = ''  # todo: add (e.g., '<some_path>/Brats18/Training')
ISIC_ORIG_DATA_DIR = ''  # todo: add (e.g., '<some_path>/isic2017-melanoma')

ISIC_BASELINE_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_baseline')
ISIC_BASELINE_MC_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_baseline_mc')
ISIC_CENTER_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_center')
ISIC_CENTER_MC_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_center_mc')
ISIC_ENSEMBLE_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_ensemble')
ISIC_AUX_FEAT_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_auxiliary_feat')
ISIC_AUX_SEGM_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_auxiliary_segm')
ISIC_ALEATORIC_PREDICT = ''  # todo: add (e.g., '<timestamp>_isic_aleatoric')

BRATS_BASELINE_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_baseline')
BRATS_BASELINE_MC_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_baseline_mc')
BRATS_CENTER_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_center')
BRATS_CENTER_MC_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_center_mc')
BRATS_ENSEMBLE_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_ensemble')
BRATS_AUX_FEAT_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_auxiliary_feat')
BRATS_AUX_SEGM_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_auxiliary_segm')
BRATS_ALEATORIC_PREDICT = ''  # todo: add (e.g., '<timestamp>_brats_aleatoric')
# directory containing all cross-validated predictions of the training set
BRATS_CV_PREDICT = ''  # todo: add (e.g., 'brats_cv_merged')


##################################################################
# Important directories. Should not but might require adaptation #
##################################################################
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')
SPLITS_DIR = os.path.join(CONFIG_DIR, 'splits')
DATASET_DIR = os.path.join(PROJECT_DIR, 'in', 'datasets')

ISIC_PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'isic_small')
ISIC_PREPROCESSED_TRAIN_DATA_DIR = os.path.join(ISIC_PREPROCESSED_DIR, 'ISIC-2017_Training')
ISIC_PREPROCESSED_TEST_DATA_DIR = os.path.join(ISIC_PREPROCESSED_DIR, 'ISIC-2017_Test_v2')

ISIC_ORIG_TRAIN_DATA_DIR = os.path.join(ISIC_ORIG_DATA_DIR, 'ISIC-2017_Training')
ISIC_ORIG_VALID_DATA_DIR = os.path.join(ISIC_ORIG_DATA_DIR, 'ISIC-2017_Validation')
ISIC_ORIG_TEST_DATA_DIR = os.path.join(ISIC_ORIG_DATA_DIR, 'ISIC-2017_Test_v2')

PREDICT_DIR = os.path.join(PROJECT_DIR, 'out', 'predictions')
ISIC_PREDICT_DIR = os.path.join(PREDICT_DIR, 'isic')
BRATS_PREDICT_DIR = os.path.join(PREDICT_DIR, 'brats')

EVAL_DIR = os.path.join(PROJECT_DIR, 'out', 'eval')
ISIC_EVAL_DIR = os.path.join(EVAL_DIR, 'isic')
BRATS_EVAL_DIR = os.path.join(EVAL_DIR, 'brats')

PLOT_DIR = os.path.join(PROJECT_DIR, 'out', 'plots')
ISIC_PLOT_DIR = os.path.join(PLOT_DIR, 'isic')
BRATS_PLOT_DIR = os.path.join(PLOT_DIR, 'brats')


####################################################################
# Definitions use in evaluation & analysis. No modification needed #
####################################################################
ECE_FOREGROUND_NAME = 'ece_foreground'
ECE_NAME = 'ece'
CALIB_NAME = 'calibration'
UNCERTAINTY_NAME = 'uncertainty'
MINMAX_NAME = 'minmax'

CALIBRATION_PLACEHOLDER = 'eval_calibration_{}.csv'
UNCERTAINTY_PLACEHOLDER = 'eval_uncertainty_{}_th{}.csv'
ECE_PLACEHOLDER = 'eval_ece_{}.csv'
MINMAX_PLACEHOLDER = 'eval_summary_minmax_{}.csv'

