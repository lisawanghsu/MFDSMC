# MFDSMC

We developed a novel machine learning framework, **MFDSMC**, for predicting cancer driver synonymous mutations. MFDSMC does not included other tool's prediction as features. We used the same  benchmark datasets  as PredDSMC and enriched the features encoded with [synMall](https://bioinfo.ahu.edu.cn/synMall/#/home). The evaluation results on benchmark test datasets demonstrate that MFDSMC outperforms existing methods, making it a valuable tool for researchers in identifying functional synonymous mutations in cancer.
The details are summarized as follows.

* data: it contains the original mutation data in folder `./vcf` and the processed data used in the project in folder `./feature-encoded`.
* code: it contains the code used in the project, including the process of training and testing the model.
  
  - for_test.py：load the model and test the data.
  - utils.py: functions used in the project.
* model: models used and saved during the project. The final model is `model\xgb_clf_mod_feat73.model`.
* results: feature importance and the optimal feature subset information.

## Environment setup

We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/).

First create a conda environment for the project:`conda create -n mfdsmc python=3.11`.

Then activate the environment:`conda activate mfdsmc`.

Then install the required packages: `pip install -r requirements.txt`.

## Usage

Please see the template data at `/data` ,it contains various characteristic data and synonymous mutations in the form of VCF. If you are trying to using MFDSMC with your own data, please process you data into the same format as it.

## Examples
If you haved encoded your mutations with the aforementioned features, you can run the following command to test the model: `python for_test.py`.
