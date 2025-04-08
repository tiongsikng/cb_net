<h1 align="center">
    Conditional Biometrics for Periocular and Face Images
</h1>
<h3 align="center">
    Conditional Multimodal Biometrics Embedding Learning For Periocular and Face in the Wild (ICPR 2022) </br>
    <a href="https://ieeexplore.ieee.org/abstract/document/9956636/"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> </br>
    On the Representation Learning of Conditional Biometrics for Flexible Deployment (IEEE Access 2023) </br>
    <a href="https://ieeexplore.ieee.org/abstract/document/10201879"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a>
</h3>
<br/>

![Network Architecture](CB_Net_Architecture.jpg?raw=true "CB-Net")
<br/></br>

## Pre-requisites:
- <b>Environment: </b>Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command for the environment requirement. These files are slightly filtered manually, so there may be redundant packages.
- <b>Dataset: </b> Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.
Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory by changing main path.
- <b>Pre-trained models: </b>(Optional) The pre-trained MobileFaceNet model for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/scl/fi/zkbuaaun22alzexw6km0x/MobileFaceNet_AF_S30.0_M0.4_D512_EP16.pth?rlkey=4b1ttgnv40tjg5x34n1hchzx7&st=b7y8etjq&dl=0).

## Training: 
1. Change hyperparameters accordingly in `params.py` file. The set values used are the default, but it is possible to alternatively change them when running the python file.
2. Run `python training/main.py`. The training should start immediately.
3. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

## Testing:
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc. 
2. Evaluation:
    * Identification / Cumulative Matching Characteristic (CMC) curve: Run `cmc_eval_identification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate CMC graph.
    * Verification / Receiver Operating Characteristic (ROC) curve: Run `roc_eval_verification.py`. Based on the generated `.pt` files in `data` directory, run `plot_cmc_roc_sota.ipynb` to generate ROC graph.
3. Visualization;
    * Decidability Index: Run `decidability_index.ipynb` to get the histogram distribution. The decidability index $d'$ will be displayed on the graph.
    * Pearson Correlation: Run `pearson_correlation.ipynb` to get the correlation curve between periocular and face. The correlation coefficient will be displayed on the graph.

## Comparison with State-of-the-Art (SOTA) models (Periocular)

| Method | Rank-1 IR (%) <br> (Periocular) | Rank-1 EER (%) <br> (Periocular) | Cross-Modal IR (%) <br> (Periocular Gallery) | Cross-Modal EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- | --- |
| PF-GLSR <a href="https://ieeexplore.ieee.org/document/9159854"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/gc7lnp66p706ecfr3exz2/AF6Jx_LKAeDOaKqDr2rbtMk?rlkey=skqp1kbwrd3uua1fk68qgmu01&st=dyunrk9r&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 79.03 | 15.56 | - | - |
| <a href="https://github.com/tiongsikng/cb_net" target="_blank" rel="noopener noreferrer"><img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="20" height="20">CB-Net</a> <a href="https://ieeexplore.ieee.org/abstract/document/10201879"> <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link"> </a> <br> <a href="https://www.dropbox.com/scl/fo/h5tz21big39wd0dzc70ou/AOabrddckd5cKUF3R2p3jw0?rlkey=l8fksw4ekat5jzcgn66jft6n3&st=t1rayruv&dl=0"> <img src="https://img.shields.io/badge/pre--trained%20weights-8A2BE2" alt="Pre-trained Weights"> </a> | 86.96 | 9.62 | 77.26 | 9.80 |

### The project directories are as follows:
<pre>
├── configs: Contains configuration files and hyperparameters to run the codes
│   ├── datasets_config.py - Contains directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., <code>/home/cb_net/data</code> (without slash).
│   └── params.py - Hyperparameters and arguments for training.
├── data: Directory for dataset preprocessing, and folder to insert data based on <code>config.py</code> files.
│   ├── <i><b>[INSERT DATASET HERE.]</i></b>
│   ├── <i>The <code>.pt</code> files to plot the CMC and ROC graphs will be generated in this directory.</i>
│   └── data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in <code>params.py</code>.
├── eval: Evaluation metrics - Identification and Verification, also contains <code>.ipynb</code> files to plot CMC and ROC graphs.
│   ├── cmc_eval_identification.py - Evaluates Rank-1 Identification Rate (IR) and generates Cumulative Matching Characteristic (CMC) curve, which are saved as <code>.pt</code> files in <code>data</code> directory. Use these <code>.pt</code> files to generate CMC curves.
│   ├── decidability_index.ipynb - Notebook to plot Decidability Index (<i>d'</i>) histogram.
│   ├── identification.py - Rank-1 Identification Rate (IR) evaluation.
│   ├── lbp_extract.py - Local Binary Pattern (LBP) feature extraction and identification calculation.
│   ├── pearson_correlation.ipynb - Notebook to plot Pearson Correlation (<i>&rho;</i>) graph.
│   ├── plot_cmc_roc.ipynb - Notebook to plot CMC and ROC side-by-side simulatenously.
│   └── roc_eval_verification.py - Evaluates Verification Equal Error Rate (EER) and generates Receiver Operating Characteristic (ROC) curve, which are saved as <code>.pt</code> files in <code>data</code> directory. Use these <code>.pt</code> files to generate ROC curves.
├── graphs: Directory where graphs and visualization evaluations are generated.
│   └── <i>CMC and ROC curve file is generated in this directory. Some evaluation images are also generated in this directory.</i>
├── logs: Directory where logs are generated.
│   └── <i>Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used.</i>
├── models: Directory to store pretrained models, and also where models are generated.
│   ├── <i><b>[INSERT PRE-TRAINED MODELS HERE.]</i></b>
│   ├── <i><b>The base MobileFaceNet for fine-tuning the CB-Net can be downloaded in <a href="https://www.dropbox.com/scl/fi/ttdt7k6ksrwjcdaj4gou7/MobileFaceNet_AF_S30.0_M0.4_D512_EP16.pth?rlkey=nybylhj1c9bf2a6i3kcp9hcc8&st=vyzrvn03&dl=0">this link</a>.</i></b>
│   └── <i>Trained models will be generated in this directory.</i>
├── network: Contains loss functions, and network related files.
│   ├── cb_net.py - CB-Net model file.
│   ├── load_model.py - Loads pre-trained weights based on a given model.
│   ├── logits.py - Contains loss functions.
│   └── mobilefacenet.py - MobileFaceNet model file.
├── <i>training:</i> Main files for training.
│   ├── main.py - Main file to run for training. Settings and hyperparameters are based on the files in <code>configs</code> directory.
│   └── train.py - Training file that is called from `main.py`. Gets batch of dataloader and contains criterion for loss back-propagation.
└── utils: Miscellaneous utility functions.
    ├── utils_cb_net.py - Conditional Biometrics (CB) Loss function (IEEE Access).
    ├── utils_cmb_net.py - Conditional Multimodal Biometrics (CMB) Loss function (ICPR).
    └── utils.py - Utility functions.
</pre>

#### Citation for this work:
<b>CMB-Net:</b>
```
@INPROCEEDINGS{cmb_net,
  author={Ng, Tiong-Sik and Low, Cheng-Yaw and Long Chai, Jacky Chen and Beng Jin Teoh, Andrew},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Conditional Multimodal Biometrics Embedding Learning For Periocular and Face in the Wild}, 
  year={2022},
  volume={},
  number={},
  pages={812-818},
  keywords={Training;Biometrics (access control);Face recognition;Biological system modeling;Neural networks;Transforms;Network architecture;Conditional Biometrics;Multimodal Biometrics;Periocular;Face;Deep Embedding Learning},
  doi={10.1109/ICPR56361.2022.9956636}}
```
<b>CB-Net:</b>
```
@ARTICLE{cb_net,
  author={Ng, Tiong-Sik and Low, Cheng-Yaw and Chai, Jacky Chen Long and Teoh, Andrew Beng Jin},
  journal={IEEE Access}, 
  title={On the Representation Learning of Conditional Biometrics for Flexible Deployment}, 
  year={2023},
  volume={11},
  number={},
  pages={82338-82350},
  keywords={Biometrics (access control);Face recognition;Correlation;Representation learning;Performance gain;Iris recognition;Conditional biometrics;face;flexible matching;periocular;representation learning},
  doi={10.1109/ACCESS.2023.3301150}}
```