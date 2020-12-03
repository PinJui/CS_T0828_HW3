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

## Results

## Reference
