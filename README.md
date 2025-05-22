# person_detect

## Installation


The code was tested on Ubuntu 20.04 and Windows 10, with [Anaconda](https://www.anaconda.com/download) Python 3.8 and [PyTorch]((http://pytorch.org/)) v1.6.0. NVIDIA GPUs are needed for both training and testing.

After install Anaconda:

0. [Optional but recommended] Create a new conda environment. 

    ~~~
    conda create --name senior python=3.8
    ~~~
    And activate the environment.
    
    ~~~
    conda activate senior
    ~~~
   Install cudatoolkit.
   ~~~
   conda install -c anaconda cudatoolkit=10.1
   ~~~

1. Install pytorch 1.6.0:

    ~~~
    conda install pytorch=1.6.0 torchvision=0.7.0 -c pytorch
    ~~~
   or
    ~~~
    pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    ~~~
    
2. Install python package requirements:

    ~~~
    pip install -r requirements.txt
    ~~~


## Inference
1. Open the "inference.py" and edit it (255 line):
   ~~~
   OP_MODE   = 'image'
   IMG_PATH  = "./data/test/"                ### input path
   OUT_PATH  = "./Output/Test/200824/test/"  ### output path
   ~~~
2. Run the "inference.py":
   ~~~
   python inference.py
   ~~~
