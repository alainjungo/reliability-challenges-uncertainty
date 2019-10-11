# Assessing Reliability and Challenges of Uncertainty Estimations for Medical Image Segmentation


This is the code to the MICCAI 2019 contribution:

"Assessing Reliability and Challenges of Uncertainty Estimations for Medical Image Segmentation" (Alain Jungo and Mauricio Reyes)



## Installation

1. Install the required Python packages (we use Python version 3.6):

   ```bash
   pip install -r requirements.txt
   ```

   

2. Get the BraTS 2018 and the ISIC 2017 dataset



## Step-by-step instruction

1. Adapt the dataset paths in the `./rechun/directories.py` file before you crate the datasets and run the training. 
2. Run the preparation scripts (`./scripts`)
3. Run the training scripts (`./bin-dl/train*`). 
   - (For *auxiliary feat.* this requires setting *others/model_dir* in the `./config/train_*_auxiliary_feat.yaml` configuration files )
4. Add the the model paths to the test configuration files (`./config`, *model_dir* entry)
5. Run the test scripts (`./bin-dl/test*`)
6. Add the prediction paths to  the `./rechun/directories.py` file
7. Run the evaluation script (`./bin-eval/eval_uncertainty.py`)
8. Run the analysis scripts (`./bin-analysis`)



#### Special case: *auxiliary segm.*

Training *auxiliary segm.* requires additional steps due to the cross-validation of the training set. *Baseline* predictions are required.

1. Train all the baseline_cv models (run `./bin-dl/*_train_default.py -config_id cv[0-4]`)

2. Add the the model paths to the test configuration files (`./config`)

3. Run the corresponding testing scripts

4. Manually copy all CV predictions into one prediction folder

5. Prepare auxiliary_segm training and prediction:

   - BraTS: 
     - add the common folder to `./rechun/directories.py` (*BRATS_CV_PREDICT*) 
     - run  `./splits/create_brats18_dataset.py --type train_with_predictions`
     - add the baseline prediction folder to `./rechun/directories.py` (BRATS_BASELINE_PREDICT) 
     - run  `./splits/create_brats18_dataset.py --type test_with_predictions`

   - ISIC:  
     - add the common folder to  `./config/train_isic_auxiliary_segm.yaml`(*others/prediction_dir*)
     - add the baseline prediction folder to  `./config/test_isic_auxiliary_segm.yaml`(*others/prediction_dir*)

6. Continue at "step-by-step instructions" step 3 (run training script)



## Structure of the project

The code consists of code for data preparation, training and testing of the models, and evaluation scripts.

The project is structured as follows:

- **Data preparation**

  - `./scripts`: data preparation scripts for the BraTS 2018 and ISIC 2017 datasets
  - The prepared data will be stored in `./in/datasets`

- **Training and testing**

  - `./bin-dl`: training and testing scripts (using the *.yaml* configuration files in `./config`)
  - `./config`: *.yaml* configuration files for the different training and testing runs
  - `./config/splits`: split files defining the training/validation/test splits
  - The models will be stored in  `./out/brats` and `./out/isic`
  - The prediction output needs to be defined in the test configuration files `./config`

- **Evaluation & Analysis**

  - `./bin-eval`: scripts that extract the information from the test images
  
  - `./bin-analysis`: scripts for analysis and plotting
  
  - The evaluation *.csv* files will be saved in `./out/eval` and the plots in `./out/plots`. 
  
    

The `./rechun/dl`, `./rechun/eval`, `./rechun/analysis` folders contain code that supports the training/testing and evaluation/analysis.

The `./common` folder contains code that is used throughout the code and could be used in other projects.



## Notes

The code runs on GPU. If you would like to run on CPU, you will need to change the *device* parameter string parameter passed to the `Context` classes in the training and testing scripts (`./bin-dl`). 

