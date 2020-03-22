# HMM-DNN Network (Speech-Recognition)

This repository is a Python implementation for HMM-DNN model which is a deep learning model in speech recognition. First, we use HMM-GMM model for labeling an existing speech data. Then, we would use this labeled data for training the HMM-DNN model. Also, we use MLP as for the DNN part of the model.



## Getting Started

### Installation

Clone the program.

`git clone https://github.com/raminnakhli/HMM-DNN-Speech-Recognition.git`



### Prerequisites

The requirements are some common packages in machine learning. You can install them using below command.

`pip install -r requirement.txt`



### Data Set

We use an Arabic speech dataset which is available on https://archive.ics.uci.edu/ml/datasets/Spoken+Arabic+Digit. For cloning the dataset, you can use the `download_data.sh` file. This file creates a data directory and downloads the dataset in it.



## Execution

Now, you can run the model with default configuration using the below command.

`python main.py`



## Contact

Ramin Ebrahim Nakhli - raminnakhli@gmail.com

Project Link: https://github.com/raminnakhli/HMM-DNN-Speech-Recognition

