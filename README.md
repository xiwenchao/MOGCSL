# Goal-Conditional Supervised Learning for Multi-task Recommendation


# Environment Requirement
The code has been tested running under Python 3.7.1. To install the required packages, run the following command:
```
conda create -n GCSL python=3.7.1

conda activate GCSL

pip install torch===1.0.1 torchvision===0.2.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

conda install cudatoolkit==10.0.130

conda install cudnn==7.6.5
```


# Command
To run the code, please use the command under the corresponding folder.

Training and testing of MOGCSL:
```
python SASRec_GSL.py
```

Training and testing of MOPRL:
```
python SASRec_PRL.py
```











