# M3d-CAM
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-available-blue.svg)](https://meclabtuda.github.io/M3d-Cam/)
[![PyPI version](https://badge.fury.io/py/medcam.svg)](https://badge.fury.io/py/medcam) 
![Python package](https://github.com/MECLabTUDA/M3d-Cam/workflows/Python%20package/badge.svg)

M3d-CAM is an easy to use Pytorch library that allows the generation of **3D/ 2D attention maps** for both **classification and segmentation** with multiple methods such as Guided Backpropagation, 
Grad-Cam, Guided Grad-Cam and Grad-Cam++. <br/> 
All you need to add to your project is a **single line of code**: <br/> 
```python
model = medcam.inject(model, output_dir="attention_maps", save_maps=True)
```

## Features

* Works with classification and segmentation data / models
* Works with 2D and 3D data
* Supports Guided Backpropagation, Grad-Cam, Guided Grad-Cam and Grad-Cam++
* Attention map evaluation with given ground truth masks
* Option for automatic layer selection

## Installation
* Install Pytorch from https://pytorch.org/get-started/locally/
* Install M3d-CAM via pip with: `pip install medcam`

## Documentation
M3d-CAM is fully documented and you can view the documentation under: <br/> 
https://meclabtuda.github.io/M3d-Cam/

## Examples

|                                            |                #1 Classification (2D)                 |                  #2 Segmentation (2D)                 |                       #3 Segmentation (3D)            |
| :----------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: |
|                  Image                     |        ![](examples/images/class_2D_image.jpg)        |        ![](examples/images/seg_2D_image.jpg)          |        ![](examples/images/seg_3D_image.jpg)          |
|          Guided backpropagation            |        ![](examples/images/class_2D_gbp.jpg)          |        ![](examples/images/seg_2D_gbp.jpg)            |        ![](examples/images/seg_3D_gbp.jpg)            |
|                 Grad-Cam                   |        ![](examples/images/class_2D_gcam.jpg)         |        ![](examples/images/seg_2D_gcam.jpg)           |        ![](examples/images/seg_3D_gcam.jpg)           |
|              Guided Grad-Cam               |        ![](examples/images/class_2D_ggcam.jpg)        |        ![](examples/images/seg_2D_ggcam.jpg)          |        ![](examples/images/seg_3D_ggcam.jpg)          |
|               Grad-Cam++                   |        ![](examples/images/class_2D_gcampp.jpg)       |        ![](examples/images/seg_2D_gcampp.jpg)         |        ![](examples/images/seg_3D_gcampp.jpg)         |

## Usage

```python
# Import M3d-CAM
from medcam import medcam

# Init your model and dataloader
model = MyCNN()
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Inject model with M3d-CAM
model = medcam.inject(model, output_dir="attention_maps", save_maps=True)

# Continue to do what you're doing...
# In this case inference on some new data
model.eval()
for batch in data_loader:
    # Every time forward is called, attention maps will be generated and saved in the directory "attention_maps"
    output = model(batch)
    # more of your code...
```

## Demos

### Classification
You can find a Jupyter Notebook on how to use M3d-CAM for classification using a resnet152 at `demos/Medcam_classification_demo.ipynb`.

### 2D Segmentation
TODO

### 3D Segmentation
You can find a Jupyter Notebook on how to use M3d-CAM with the nnUNet for handeling 3D data at `demos/Medcam_nnUNet_demo.ipynb`.
