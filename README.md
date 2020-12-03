# CS_T0828_HW3
Instance Segmentation On Tiny Pascal Dataset

## Content

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Results](#results)
- [Reference](#reference)

## Introduction
This homework requires to train a model to perform instance segmentation on the given image.

The given dataset has 1349 training images with 20 common object classes and 100 test image for inference.

Check some examples in [train]() and [test]().

The desired output is a list of dict, which length of the list is thte number of detected instances.

And each dict should contain keys below:

- "image_id": id of test image, which is the key in “test.json”, int
- “score”: probability for the class of this instance, float
- “category_id”: category id of this instance, int
- “segmentation”: Encode the mask in Run Length Encoding by provide function, str

Result Example(For the first instance):
```
{
"image_id": 914,
"category_id": 3,
"segmentation": {"size": [333, 500], "counts": "dZ[13R::I7K7J3M3N101N1O2O0O2N1N3N2N1100O10000O10O1O010O1O1O1O1O1O1O010O010O010O01O01O010O100000000000O100O10001O0O02OO10O10O0100O10O01000000O01000O010O010UIaNg4^1XKQOZ4P1dKRO\\4n0bKTO^4l0`KUOa4k0]KWOb4j0\\KXOd4h0ZKXOh4h0TKZOn4f0jJAW5?gJA[5?cJA^5`0_JBb5=]JCe5?WJAk5c0oI]OR6h0gIXO\\6[2101N101N100O2M2O1N2O1O2N1O1N2N2O1N2O1O2N1O001O100O00100O010O1N1O100O101O0O110O0001O0001O000O101N1O2N1N3M3N1O2O1O001N1O1N3M2O3L3M3M4M2N2N2N3M2N3M2N3M3K6JQdP2"}, 
"score": 0.9939279556274414
}
```
## Methodology

Here I use the toolbox Detectron2<a href="#[1]"> [1] </a> on Colab to train the model.

### Step 1: Install the Detectron2 toolbox

First `%cd` to your working directory.

Install the Detectron toolbox at your working directory by running the commands below:

```
!pip install -U torch torchvision
!pip install git+https://github.com/facebookresearch/fvcore.git
!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
!pip install -e detectron2_repo
```

After install is complete, remember to press the **Restart Runtime** button to make the installation to take effect.

### Step 2: Register the dataset and create a metadata

Register your dataset by running the commands below

```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("tiny_voc", {}, "/Your/Train/ImageFolder/Path")
```
and create a metadeta to tell the dataset info by commands

```python
from detectron2.data import DatasetCatalog, MetadataCatalog
tiny_voc_metadata = MetadataCatalog.get("tiny_voc")
tiny_voc_metadata
```
You can check the classes names by `tiny_voc_metadata.thing_classes`.

Also you can check the training image and the masks by

```python
import random
import cv2
from google.colab.patches import cv2_imshow
from detectron2.utils.visualizer import Visualizer

dataset_dicts = DatasetCatalog.get("tiny_voc")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=tiny_voc_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])
```

## Step 3: Start Training

First setup the config file and start training by:

```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(
    "./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("tiny_voc",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0003
cfg.SOLVER.MAX_ITER = (
    10000
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

You can set the parameters to meet your task requirement.

And to resume the training by setting the resume parameter to `True`

The default OUTPUT_DIR is `./output`

You should see the training begins.

## Step 4:Inference

Create another same config to avoid confusing with training config.

```python
testcfg = get_cfg()
testcfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
testcfg.DATASETS.TRAIN = ("tiny_voc",)
testcfg.DATASETS.TEST = ("tiny_voc")
testcfg.DATALOADER.NUM_WORKERS = 2
testcfg.SOLVER.IMS_PER_BATCH = 2
testcfg.SOLVER.BASE_LR = 0.001
testcfg.SOLVER.MAX_ITER = 500
testcfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
testcfg.MODEL.ROI_HEADS.NUM_CLASSES = 20 
testcfg.MODEL.WEIGHTS = os.path.join(testcfg.OUTPUT_DIR, "model_final.pth")
testcfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
```

The things to modify here are your `testcfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST` to select the score threshold and `testcfg.MODEL.WEIGHTS` to select the desired inferencing weight file.

Create the predictor

```python
from detectron2.engine import DefaultPredictor
predictor = DefaultPredictor(testcfg)
```

Visualize some result(21 images):

```python
from itertools import groupby
from pycocotools.coco import COCO
from pycocotools import mask as maskutil
from detectron2.utils.visualizer import ColorMode
import cv2
from google.colab.patches import cv2_imshow
from detectron2.utils.visualizer import Visualizer
coco_test = COCO("Your/test.json/file/path")
counter = 0
for imgid in coco_test.imgs:
  image = cv2.imread("Your/Test/image/folder/" + coco_test.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1] # load image
  outputs = predictor(image)
  v = Visualizer(image,
                      metadata=tiny_voc_metadata,
                      scale=1.2, 
                      instance_mode=ColorMode.IMAGE_BW
          )
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2_imshow(v.get_image()[:,:,::-1])
  if counter == 20:
    break
  counter +=1
```

## Step 5: Generate output file:
```python
coco_test = COCO("Your/test.json/file/path")

coco_output = []
import json

for imgid in coco_test.imgs:
    image = cv2.imread("Your/Test/image/folder/" + coco_test.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1] # load image
    outputs = predictor(image)
    for i_instance in range(len(outputs["instances"].scores)):
        pred = {}
        pred['image_id'] = imgid
        pred['category_id'] = int(outputs["instances"].pred_classes[i_instance] + 1)
        binary_mask = outputs["instances"].pred_masks[i_instance].to("cpu").numpy()

        pred['segmentation'] = binary_mask_to_rle(binary_mask)
        pred['score'] = float(outputs["instances"].scores[i_instance])
        coco_output.append(pred)
with open('Your/output/file/.json', "w") as f:
    json.dump(coco_output, f)
```

## Results
Images:

![result1](https://github.com/PinJui/CS_T0828_HW3/blob/main/assets/hw3result.png)

![result2](https://github.com/PinJui/CS_T0828_HW3/blob/main/assets/hw3result2.png)

![result3](https://github.com/PinJui/CS_T0828_HW3/blob/main/assets/hwresult3.png)

Json file:
```
{
"image_id": 914, 
"category_id": 3, 
"segmentation": 
    {
"size": [333, 500],
"counts":"dZ[13R::I7K7J3M3N101N1O2O0O2N1N3N2N1100O10000O10O1O010O1O1O1O1O1O1O010O010O010O01O01O010O100000000000O100O10001O0O02OO10O10O0100O10O01000000O01000O010O010UIaNg4^1XKQOZ4P1dKRO\\4n0bKTO^4l0`KUOa4k0]KWOb4j0\\KXOd4h0ZKXOh4h0TKZOn4f0jJAW5?gJA[5?cJA^5`0_JBb5=]JCe5?WJAk5c0oI]OR6h0gIXO\\6[2101N101N100O2M2O1N2O1O2N1O1N2N2O1N2O1O2N1O001O100O00100O010O1N1O100O101O0O110O0001O0001O000O101N1O2N1N3M3N1O2O1O001N1O1N3M2O3L3M3M4M2N2N2N3M2N3M2N3M3K6JQdP2"
    }, 
"score": 0.9939279556274414
}
```

## Reference

<a name="[1]"> [1] [Detectron2](https://github.com/facebookresearch/detectron2)</a>

[2] [How to train Detectron2 with Custom COCO Datasets](https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/)

[3] [绘制COCO数据集结果](https://www.w3xue.com/exp/article/201811/8175.html)
