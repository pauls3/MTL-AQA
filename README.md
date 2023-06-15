# MTL-AQA
[What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/abs/1904.04346)

<b>***</b> <i>Want to know the score of a Dive at the ongoing Olympics, even before the judges' decision?</i> <b>Try out our [AI Olympics Judge](https://share.streamlit.io/gitskim/aqa_streamlit/main/main.py) ***</b>


## MTL-AQA Concept:

<p align="center"> <img src="diving_sample.gif?raw=true" alt="diving_video" width="200"/> </p>
<p align="center"> <img src="mtlaqa_concept.png?raw=true" alt="mtl_net" width="400"/> </p>


## Installation Instructions
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

[Download the Sports-1M pretrained C3D weights.](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle) Save the pickle file under `code_release/`. So it should be located in `MTL-AQA/code_release/c3d.pickle`.


## Dataset
The dataset should be available in the AI server under Dr. Morris's code. Please do not delete it.


## Running the Code
### Terminal Instance Using Screen
Before running any code, please make a terminal instance using the screen command. For our case, let's make a screen with the following command:
```
screen -S aqa
```

This will create an instance of the terminal that you can go back on after logging off with `ctrl+d`. You can exit the terminal instance with `ctrl+d`. You can enter back into the screen with the following command:
```
screen -r aqa
```

If you forget what your screen name was, use the following command to list all of your screens:
```
screen -r
```

While working on the AI Server, its best to run any code in a screen instance like above.



### GPU Availability
In the `code_release` directory, please check what GPUs are available in the AI Server with the following command:
```
nvidia-smi
```

You should get results similar to this:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.57       Driver Version: 450.57       CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro RTX 6000     On   | 00000000:1A:00.0 Off |                  Off |
| 33%   28C    P8    15W / 260W |      0MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Quadro RTX 6000     On   | 00000000:1C:00.0 Off |                  Off |
| 33%   43C    P2    68W / 260W |   1658MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Quadro RTX 6000     On   | 00000000:1D:00.0 Off |                  Off |
| 33%   31C    P8    19W / 260W |     27MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Quadro RTX 6000     On   | 00000000:1E:00.0 Off |                  Off |
| 33%   33C    P8    15W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  Quadro RTX 6000     On   | 00000000:3D:00.0 Off |                  Off |
| 33%   28C    P8    12W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  Quadro RTX 6000     On   | 00000000:3F:00.0 Off |                  Off |
| 33%   29C    P8    14W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  Quadro RTX 6000     On   | 00000000:40:00.0 Off |                  Off |
| 33%   39C    P8    12W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  Quadro RTX 6000     On   | 00000000:41:00.0 Off |                  Off |
| 33%   31C    P8    15W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```


Note the GPU numbers are [0-7]. Select a GPU that is not being used, which is based on the memory usage column in the middle. As we can see here, GPU 1 is being used (1658MiB / 24220MiB). You want to select a GPU with like 10 or less MiB being used, or else you will run into an error or potentially kick someone off from that GPU.

Let us choose GPU 6 for our example. Using the following command `CUDA_VISIBLE_DEVICES="6"` before running any CUDA script will only make GPU 6 be visible to the script. In our case, run the following command:

```
CUDA_VISIBLE_DEVICES="6" python3 train_test_C3DAVG.py
```

It should result in something like:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.57       Driver Version: 450.57       CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro RTX 6000     On   | 00000000:1A:00.0 Off |                  Off |
| 33%   28C    P8    16W / 260W |      0MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Quadro RTX 6000     On   | 00000000:1C:00.0 Off |                  Off |
| 33%   44C    P2    69W / 260W |   1658MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Quadro RTX 6000     On   | 00000000:1D:00.0 Off |                  Off |
| 33%   31C    P8    19W / 260W |     27MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Quadro RTX 6000     On   | 00000000:1E:00.0 Off |                  Off |
| 33%   33C    P8    15W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  Quadro RTX 6000     On   | 00000000:3D:00.0 Off |                  Off |
| 33%   29C    P8    11W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  Quadro RTX 6000     On   | 00000000:3F:00.0 Off |                  Off |
| 33%   30C    P8    15W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  Quadro RTX 6000     On   | 00000000:40:00.0 Off |                  Off |
| 33%   41C    P2    83W / 260W |   5234MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  Quadro RTX 6000     On   | 00000000:41:00.0 Off |                  Off |
| 33%   31C    P8    14W / 260W |      3MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    6   N/A  N/A     47119      C   python3                          5231MiB |
+-----------------------------------------------------------------------------+
```

Here, you can see that GPU 6 is running the job by only using 5MiB.

You can also use multiple GPUs if needed (e.g., bigger batch size/model) with by seperating GPU numbers with commas:
```
CUDA_VISIBLE_DEVICES="4,6,7"
```

### Training the C3D Model
Use the following command to train a C3D model from scratch on the AI server:
```
CUDA_VISIBLE_DEVICES="<[0-7]>" python3 train_test_C3DAVG.py
```

Where [0-7] denotes the choice of a GPU. Currently with how the code is setup, 1 GPU is suffecient.


##
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
