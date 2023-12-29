# Conditional Biometrics for Periocular and Face Images
## Conditional Multimodal Biometrics Embedding Learning For Periocular and Face in the Wild (ICPR 2022)
## On the Representation Learning of Conditional Biometrics for Flexible Deployment (IEEE Access 2023)

This repository contains codes for 2 papers:
* Conditional Multimodal Biometrics Embedding Learning For Periocular and Face in the Wild (International Conference on Pattern Recognition (ICPR)) [(Paper)](https://ieeexplore.ieee.org/abstract/document/9956636/)
* On the Representation Learning of Conditional Biometrics for Flexible Deployment (IEEE Access) [(Paper)](https://ieeexplore.ieee.org/abstract/document/10201879)

![Network Architecture](CB_Net_Architecture.jpg?raw=true "CB-Net")


The project details are as follows:

- configs: Contains configuration files and hyperparameters to run the codes
    * config.py - Contains directory path for dataset files. Change 'main' in 'main_path' dictionary accordingly.
    * params.py - Hyperparameters and arguments for training.
- data: Directory for dataset preprocessing, and folder to insert data based on `config.py` files.
    * __**INSERT DATASET HERE**__
    * _CMC and ROC data dictionaries are generated in this directory._
    * data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in `params.py`.
- eval: Evaluation metrics - Identification and Verification, also contains `.ipynb` files to plot CMC and ROC graphs.
    * cmc_eval.py - Cumulative Matching Characteristic (CMC) evaluation, also saves `.pt` files for plotting CMC.
    * identification.py - Rank-1 Identification Rate (IR) evaluation.
    * pearson_correlation.ipynb - Plot Pearson Correlation graph.
    * plot_cmc.ipynb - Notebook to plot CMC curve.
    * plot_cmc_roc.ipynb - Notebook to plot CMC and ROC side-by-side simulatenously.
    * plot_histogram.py - Plot <i>d'</i> histogram for inter-modal/intra-modal matching.
    * plot_roc.ipynb - Notebook to plot ROC curve.
    * roc_eval.py - Receiver Operating Characteristic (ROC) evaluation, also saves `.pt` files for plotting ROC.
    * verification.py - Verification Equal Error Rate (EER) evaluation.
- graphs: Directory where graphs are generated.
    * _CMC and ROC curve file is generated in this directory._
- logs: Directory where logs are generated.
    * _Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used._
- models: Directory to store pretrained models, and also where models are generated.
    * __**INSERT PRE-TRAINED MODELS HERE. The base MobileFaceNet for fine-tuning the CB-Net can be downloaded in [this link](https://www.dropbox.com/scl/fo/58uzvjul7g0n77m66hv61/h?rlkey=vw5bi0ipm054tfwxsm5b8no9g&dl=0).**__
    * _Trained models will also be stored in this directory._
- network: Contains loss functions, and network related files.
    * cb_net.py - CB-Net model file (ICPR/IEEE Access).
    * load_model.py - Loads pre-trained weights based on a given model.
    * logits.py - Contains loss functions.
    * mobilefacenet.py - MobileFaceNet model file.
- __training:__ Main files for training.
    * main.py - Main file to run for training. Settings and hyperparameters are based on the files in `configs` directory.
    * train.py - Training file that is called from `main.py`. Gets batch of dataloader and contains criterion for loss back-propagation.
- utils: Miscellaneous utility functions.
    * utils_cb_net.py - Conditional Biometrics (CB) Loss function (IEEE Access).
    * utils_cmb_net.py - Conditional Multimodal Biometrics (CMB) Loss function (ICPR).
    * utils.py - Utility functions.

### Pre-requisites (requirements):
Check `environment.yml` file, which was generated using `conda env export > environment.yml --no-builds` command. Else, check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command. These files are not filtered, so there may be redundant packages.
Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.

### Training:
1. Ensure that datasets are located in `data` directory. Configure `config.py` file to point to this data directory.
2. Change hyperparameters accordingly in `params.py` file. The set values used are the default.
3. Run `main.py` file. The training should start immediately.
4. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

### Testing:
0. Pre-trained models for fine-tuning can be downloaded from [this link](https://www.dropbox.com/s/g8gn4x4wp0svyx5/pretrained_models.zip?dl=0). Password is _conditional\_biometrics_.
1. Based on the (pre-)trained models in the `models` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc.
2. Evaluation:
    * Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Identification: Run `identification.py`.
    * Receiver Operating Characteristic (ROC) curve: Run `roc_eval.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
    * Verification: Run `verification.py`.