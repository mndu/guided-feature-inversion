# Guided feature inversion
PyTorch code for paper: Towards Explanation of DNN-based Prediction with Guided Feature Inversion. It has been accepted in [KDD2018](https://www.kdd.org/kdd2018/).

We propose a guided feature inversion method to provide instance-level interpretations of CNN predictions. The proposed method could locate the salient foreground part, thus determining which part of information in the input instance is preserved by the CNN, and which part is discarded.

## Usage Instructions:
* Clone the code from Github:
```
git clone https://github.com/mndu/guided-feature-inversion.git
cd guided-feature-inversion
```


* Provide explanation to a CNN prediction for an input image. Here we use VGG-19, for both l0 and l1 layer, we use `pool5` layer. The explanation is for the prediction with the largest probability. 
```
python main.py --layer features.36 --epochs 80 --gpu 0 --network vgg19 --label 1 --image ILSVRC2012_val_00000021.JPEG
```

* By replacing the `--label` with other categories, this framework also could provide explanations for different classes of objects.

* This framework can also give explanations for other CNN architectures, such as AlexNet, and ResNet, by replacing with appropriate network `--network` and layer `--layer`.

## System requirement:
Python 2.7, torch 0.3, torchvision, matplotlib, PIL, cv2, and skimage.


## Acknowledgement:
This work is motivated by the feature inversion paper [Understanding deep image representations by inverting them](https://arxiv.org/abs/1412.0035), and the [feature inversion implementation](https://github.com/ruthcfong/invert).


## Reference:
```
@inproceedings{du2018kdd,
    author    = {Mengnan Du, Ninghao Liu, Qingquan Song, and Xia Hu},
    title     = {Towards Explanation of DNN-based Prediction with Guided Feature Inversion},
    booktitle = {The 24rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
    year      = {2018}
}
```