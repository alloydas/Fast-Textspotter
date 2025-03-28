# Harnessing the Power of Multi-Lingual Datasets for Pre-training: Towards Enhancing Text Spotting Performance


#### Installation

Please refer to the **Installation** section of AdelaiDet: [README.md](https://github.com/aim-uofa/AdelaiDet/blob/master/README.md). 

If you have not installed Detectron2, following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md). 

After that, build this repository with

```bash
python setup.py build develop
```

#### Preparing Datasets

Please download TotalText, CTW1500, MLT, and CurvedSynText150k according to the guide provided by AdelaiDet: [README.md](https://github.com/aim-uofa/AdelaiDet/blob/master/datasets/README.md).

ICDAR2015 dataset can be download via [link](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/EWgEM5BSRjBEua4B_qLrGR0BaombUL8K3d23ldXOb7wUNA?e=7VzH34).

Extract all the datasets and make sure you organize them as follows

The new Dataset is pushed into dataset folder download it, extract it, convert it and then just run it. 

```
- datasets
  | - CTW1500
  |   | - annotations
  |   | - ctwtest_text_image
  |   | - ctwtrain_text_image
  | - totaltext (or icdar2015)
  |   | - test_images
  |   | - train_images
  |   | - test.json
  |   | - train.json
  | - mlt2017 (or syntext1, syntext2)
      | - annotations
      | - images
```

After that, download [polygonal annotations](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/ES4aqkvamlJAgiPNFJuYkX4BLo-5cDx9TD_6pnMJnVhXpw?e=tu9D8t), along with [evaluation files](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/Ea5oF7VFoe5NngUoPmLTerQBMdiVUhHcx2pPu3Q5p3hZvg?e=2NJNWh) and extract them under `datasets` folder.

#### Visualization Demo

You can try to visualize the predictions of the network using the following command:

```bash
python demo/demo.py --config-file <PATH_TO_CONFIG_FILE> --input <FOLDER_TO_INTPUT_IMAGES> --output <OUTPUT_FOLDER> --opts MODEL.WEIGHTS <PATH_TO_MODEL_FILE> MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.3
```

You may want to adjust `INFERENCE_TH_TEST` to filter out predictions with lower scores.

#### Training

You can train from scratch or finetune the model by putting pretrained weights in `weights` folder.

Example commands:

```bash
python tools/train_net.py --config-file <PATH_TO_CONFIG_FILE> --num-gpus 8
```
You can train chineese from scratch or finetune the model by putting pretrained weights in `weights` folder.
```bash
 python tools/train_net.py --config-file configs/TESTR/CHN/TESTR_swin_Polygon.yaml
```

All configuration files can be found in `configs/TESTR`, excluding those files named `Base-xxxx.yaml`.

`TESTR_R_50.yaml` is the config for TESTR-Bezier, while `TESTR_R_50_Polygon.yaml` is for TESTR-Polygon.

#### Evaluation

```bash
python tools/train_net.py --config-file <PATH_TO_CONFIG_FILE> --eval-only MODEL.WEIGHTS <PATH_TO_MODEL_FILE>
```
