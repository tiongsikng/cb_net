# Conditional Biometrics for Periocular and Face Images
## Conditional Multimodal Biometrics Embedding Learning For Periocular and Face in the Wild (ICPR 2022)
## On the Representation Learning of Conditional Biometrics for Flexible Deployment (IEEE Access 2023)

This repository contains codes for 2 papers (1 conference, and 1 extended journal):
* Conditional Multimodal Biometrics Embedding Learning For Periocular and Face in the Wild (International Conference on Pattern Recognition (ICPR)) [(Paper)](https://ieeexplore.ieee.org/abstract/document/9956636/)
* On the Representation Learning of Conditional Biometrics for Flexible Deployment (IEEE Access) [(Paper)](https://ieeexplore.ieee.org/abstract/document/10201879)

![Network Architecture](CB_Net_Architecture.jpg?raw=true "CB-Net")


The project directories are as follows:

- configs: Contains configuration files and hyperparameters to run the codes
    * config.py - Contains directory path for dataset files. Change 'main' in 'main_path' dictionary accordingly.
    * params.py - Hyperparameters and arguments for training.
- data: Directory for dataset preprocessing, and folder to insert data based on `config.py` files.
    * __**INSERT DATASET HERE**__
    * _CMC and ROC data dictionaries are generated in this directory._
    * data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in `params.py`.
- eval: Evaluation metrics - Identification and Verification, also contains `.ipynb` files to plot CMC and ROC graphs.
    * cmc_eval_identification.py - Evaluates Rank-1 Identification Rate (IR) and generates Cumulative Matching Characteristic (CMC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate CMC curves.
    * decidability_index.ipynb - Notebook to plot Decidability Index ($d'$) histogram.
    * identification.py - Rank-1 Identification Rate (IR) evaluation.
    * lbp_extract.py - Local Binary Pattern (LBP) feature extraction and identification calculation.
    * pearson_correlation.ipynb - Notebook to plot Pearson Correlation ($\rho$) graph.
    * plot_cmc_roc.ipynb - Notebook to plot CMC and ROC side-by-side simulatenously.
    * roc_eval_verification.py - Evaluates Verification Equal Error Rate (EER) and generates Receiver Operating Characteristic (ROC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate ROC curves.
- graphs: Directory where graphs are generated.
    * _CMC and ROC curve graphs are generated in this directory._
- logs: Directory where logs are generated.
    * _Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used._
- models: Directory to store pretrained models, and also where models are generated.
    * __**INSERT PRE-TRAINED MODELS HERE. The base MobileFaceNet for fine-tuning the CB-Net can be downloaded in [this link](https://www.dropbox.com/scl/fi/zkbuaaun22alzexw6km0x/MobileFaceNet_AF_S30.0_M0.4_D512_EP16.pth?rlkey=4b1ttgnv40tjg5x34n1hchzx7&st=b7y8etjq&dl=0).**__
    * _Trained models will also be stored in this directory._
- network: Contains loss functions, and network related files.
    * cb_net.py - CB-Net model file.
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

### Pre-requisites:
- <b>Environment: </b>Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command for the environment requirement. These files are slightly filtered manually, so there may be redundant packages.
- <b>Dataset: </b> Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.
Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory by changing main path.
- <b>Pre-trained models: </b>(Optional) The pre-trained MobileFaceNet model for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/scl/fi/zkbuaaun22alzexw6km0x/MobileFaceNet_AF_S30.0_M0.4_D512_EP16.pth?rlkey=4b1ttgnv40tjg5x34n1hchzx7&st=b7y8etjq&dl=0).

### Training: 
1. Change hyperparameters accordingly in `params.py` file. The set values used are the default, but it is possible to alternatively change them when running the python file.
2. Run `python training/main.py`. The training should start immediately.
3. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

### Testing:
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc. 
2. Evaluation:
    * Identification / Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval_identification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Verification / Receiver Operating Characteristic (ROC) curve: Run `roc_eval_verification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
3. Visualization;
    * Decidability Index: Run `decidability_index.ipynb` to get the histogram distribution. The decidability index $d'$ will be displayed on the graph.
    * Pearson Correlation: Run `pearson_correlation.ipynb` to get the correlation curve between periocular and face. The correlation coefficient will be displayed on the graph.

### Comparison with State-of-the-Art (SOTA) models (Periocular)

| Method | Rank-1 IR (%) <br> (Periocular) | Rank-1 EER (%) <br> (Periocular) | Cross-Modal IR (%) <br> (Periocular Gallery) | Cross-Modal EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- | --- |
| [PF-GLSR](https://ieeexplore.ieee.org/document/9159854) [(Weights)](https://www.dropbox.com/scl/fo/gc7lnp66p706ecfr3exz2/AF6Jx_LKAeDOaKqDr2rbtMk?rlkey=skqp1kbwrd3uua1fk68qgmu01&st=dyunrk9r&dl=0) | 79.03 | 15.56 | - | - |
| CB-Net [(Weights)](https://www.dropbox.com/scl/fo/h3grey98yeh0ir7i82lbd/AINQZy8eAEU3F4rXJm50MCE?rlkey=h0i1vv0a36uu4xsd2s41bdnaf&st=3ws0bo5q&dl=0) | 86.96 | 9.62 | 77.26 | 9.80 |