# MTL-AQA
[What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/abs/1904.04346)

<b>***</b> <i>Want to know the score of a Dive at the ongoing Olympics, even before the judges' decision?</i> <b>Try out our [AI Olympics Judge](https://share.streamlit.io/gitskim/aqa_streamlit/main/main.py) ***</b>


## MTL-AQA Concept:

<p align="center"> <img src="diving_sample.gif?raw=true" alt="diving_video" width="200"/> </p>
<p align="center"> <img src="mtlaqa_concept.png?raw=true" alt="mtl_net" width="400"/> </p>


# Installation Instructions
This code was setup on the UNLV ECE AI server with the following environment.

Create a conda environment with Python 3.7. Install the conda packages with the following commands:

```
conda create --name mlt-aqa python=3.7
conda activate mtl-aqa

conda install pillow numpy scipy
conda install -c hcc cudatoolkit=10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```


If you are having any errors in running the code, then you will have to update numpy through pip:

```
pip install numpy --upgrade
```

[Download the Sports-1M pretrained C3D weights.](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle) Save the pickle file under 



This repository contains MTL-AQA dataset + code introduced in the above paper. If you find this dataset or code useful, please consider citing:
```
@inproceedings{mtlaqa,
  title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
  author={Parmar, Paritosh and Tran Morris, Brendan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={304--313},
  year={2019}
}
```

## Check out our other relevant works:

[Fine-grained Exercise Action Quality Assessment](https://github.com/ParitoshParmar/Fitness-AQA): Self-Supervised Pose-Motion Contrastive Approaches for Fine-grained Action Quality Assessment (can be used for Diving as well!) + Fitness-AQA dataset
