# Adversarial-Variational-Semi-supervised-Learning
## Title: Adversarial variational embedding for robust semi-supervised learning

**PDF: [KDD2019](https://dl.acm.org/doi/abs/10.1145/3292500.3330966), [arXiv](https://arxiv.org/abs/1905.02361)**

**Authors: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://www.linayao.com/) (lina.yao@unsw.edu.au), Feng Yuan**

## Overview
This repository contains reproducible codes for the proposed AVAE model.  
In this paper, we present an effective and robust semi-supervised latent representation framework, AVAE, by proposing a modified VAE model and integration with generative adversarial networks. The VAE++ and GAN share the same generator. In order to automatically learn the exclusive latent code, in the VAE++, we explore the latent code’s posterior distribution and then stochastically generate a latent representation based on the posterior distribution. The discrepancy between the learned exclusive latent code and the generated latent representation is constrained by semi-supervised GAN. The latent code of AVAE is finally served as the learned feature for classification. 
<p align="center">
<img src="https://raw.githubusercontent.com/xiangzhang1015/Adversarial-Variational-Semi-supervised-Learning/master/structure%20of%20proposed%20Adversarial-Variational-Semi-supervised-Learning.PNG", width="400", align="center">
</p>

<center><b>Structure of the proposed AVAE model framework</b></center>


## Code
[AVAE.py](https://github.com/xiangzhang1015/Adversarial-Variational-Semi-supervised-Learning/blob/master/AVAE.py) is the main file and other .py files are the related functions.


## Citing
If you find our work useful for your research, please consider citing this paper:

    @inproceedings{zhang2019adversarial,
      title={Adversarial variational embedding for robust semi-supervised learning},
      author={Zhang, Xiang and Yao, Lina and Yuan, Feng},
      booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
      pages={139--147},
      year={2019}
    }

## Requirements 
> Python == 2.7  
  Numpy == 1.11.2  
  TensorFlow == 1.3.0

## Datasets
All the datasets used in this paper are pretty large which are diffilut to upload to github. Fortunately, the datasets are public available and our paper basically used the raw data, so please access the original data in the following links.

PAMAP2: http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

TUH: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

MNIST: http://yann.lecun.com/exdb/mnist/

Yelp: https://www.yelp.com/dataset

The datasets are huge, using small subset for debugging is strongly recommended. There are very detail comments in the code in order to help understanding. 


## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang.alan.zhang@gmail.com>.


## License

This repository is licensed under the MIT License.
