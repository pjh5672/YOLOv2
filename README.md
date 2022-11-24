# <div align="center">YOLOv2</div>

---

## [Content]
1. [Description](#description)   
2. [Usage](#usage)  
2-1. [K-medoids Anchor Clustering](#k-medoids-anchor-clustering)  
2-2. [Model Training](#model-training)  
2-3. [Detection Evaluation](#detection-evaluation)  
2-4. [Result Analysis](#result-analysis)  
3. [Contact](#contact)   

---

## [Description]

This is a repository for PyTorch implementation of YOLOv2 following the original paper (https://arxiv.org/abs/1612.08242).   

 - **Performance Table**

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | mAP<br><sup>(@0.5:0.95) | mAP<br><sup>(@0.5) | Params<br><sup>(M) | FLOPS<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| YOLOv2<br><sup>(<u>Paper:page_with_curl:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 416 | 51.00 | 76.8 | *not reported* | 34.90 |
| YOLOv2-416<br><sup>(<u>Our:star:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 416 | - | - | 50.67 | 29.49 |
| YOLOv2-multi scale<br><sup>(<u>Our:star:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 416 | 48.5 | 75.2 | 50.67 | 29.49 |


<div align="center">

  <a href=""><img src=./asset/EP_150.jpg width="100%" />

</div>


## [Usage]

#### K-medoids Anchor Clustering   
 - You extract anchor box priors from all instances' boxes at first.

 ```python
python kmedoids_anchor.py --exp_name my_test --data voc.yaml
 ```


```log
2022-11-16 13:43:54 | Avg IOU: 62.01%
2022-11-16 13:43:54 | Boxes:
    [[0.068      0.11711711]
    [0.16       0.26666668]
    [0.278      0.60982656]
    [0.776      0.82133335]
    [0.494      0.40533334]]
2022-11-16 13:43:54 | Ratios: [0.46, 0.58, 0.6, 0.94, 1.22]
```

<div align="center">

  <a href=""><img src=./asset/box_hist.jpg width="40%" /></a>

</div>


#### Model Training 
 - You can train your own YOLOv2 model using Darknet-19 with anchor box from above step.

```python
python train.py --exp_name my_test --data voc.yaml
```


#### Detection Evaluation
 - You can compute detection metric via mean Average Precision(mAP) with IoU of 0.5, 0.75, 0.5:0.95. I follow the evaluation code with the reference on https://github.com/rafaelpadilla/Object-Detection-Metrics.

```python
python val.py --exp_name my_test --data voc.yaml --ckpt_name best.pt
```


#### Result Analysis
 - After training is done, you will get the results shown below.

<div align="center">

  <a href=""><img src=./asset/figure_AP.jpg width="60%" /></a>

</div>


```log
2022-11-16 13:45:37 | YOLOv2 Architecture Info - Params(M): 50.67, FLOPS(B): 29.49
2022-11-16 13:48:09 | [Train-Epoch:001] multipart: 308.6470  obj: 1.8051  noobj: 380.8712  txty: 0.8276  twth: 20.2762  cls: 10.8872  
2022-11-09 18:01:53 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.336
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.054
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.011
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.039
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.131
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.405

                                                ...

2022-11-09 22:59:17 | [Train-Epoch:135] multipart: 1.6719  obj: 0.3912  noobj: 0.3355  box: 0.1720  cls: 0.2530  
2022-11-09 22:59:40 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.605
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.292
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.036
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.113
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.293
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.399
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.688
```


<div align="center">

<a href=""><img src=./asset/car.jpg width="22%" /></a> <a href=""><img src=./asset/cat.jpg width="22%" /></a> <a href=""><img src=./asset/dog.jpg width="22%" /></a> <a href=""><img src=./asset/person.jpg width="22%" /></a>

</div>


---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  