# HW1P2: Submission Details

## Running Intructions:
The Jupyter notebook should be self contained, and the cells can be ran as is. *HW2P2_Alec.ipynb* directly took cells from *HW2P2_S24_Starter_Notebook.ipynb* with added notes and personal implementations of residual basic blocks and bottleneck blocks.  

## Experiments:

**Resources:**
1. https://docs.google.com/spreadsheets/d/1XbuWYZajM0SxMG7pE24waEGNMkX4CvrBSnlW6SH_1LU/edit#gid=0 
2. https://wandb.ai/atrela-cmu/hw2p2-ablations?nw=nwuseratrela
    *Shared with a couple of teammates, the best run of mine is **resnet-bottleneck-official** achieving slighly over 90% accuracy.*

Before beginning this assignment I read a couple of the recommended papers: The Bag of Tricks for CNNs, ResNet, and ConvNext papers. I started by making a deep ConvNet within the parameter limit by using many of the recommendations from the BOT paper, since these were most easily implemented. This included things like the cosine annealing learning rate scheduler, a linear learning rate warmup, a starting learning rate proportional to batchsize, and a label smoothing on the cross-entropy loss. This offered mild success, at least better than the early-submission architecture. 

After capping off the abilities of ConvNets alone, I decided to make a much deeper network through the use of the basic block presented in the ResNet paper. This offered absurd gains of upwards of 15%. This nearly propelled me to the high cutoff. After seeing the gains made available from increasing depth, I decided to implement the bottleneck block from scratch. I believed that depth increased the expression of the network, and the bottleneck block allowed me to increase the depth versus the basic block by a non-negligible amount. 

I modeled my bottleneck block and resnet bottleneck network based on the official PyTorch implementation. There are certain transitions that occur between of channel size of a certain length and they also implement strides at these depths to ensure that the model does not grow too large. I used 7 blocks of the following inchannels: 256, 512, & 1024. This architecture, along with the standard stem presented in the paper, was enough to propel me to my highest achieving architecture. This also got me to the high-cutoff on verification dataset as well. It was also at this point that I implemented three simple augmentations. 

## Highest Performing Model:

| Parameter              | Value                                                                                                   | Description                                                               |
|------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `batch_size`           | 128                                                                                                     | Increase this if your GPU can handle it                                   |
| `lr`                   | 0.02                                                                                                    |                                                                           |
| `epochs`               | 100                                                                                                     | 20 epochs is recommended ONLY for the early submission                    |
| `warmstart`            | 15                                                                                                      |                                                                           |
| `label-smoothing`      | 0.1                                                                                                     |                                                                           |
| `model`                | resnet-bottleneck-check                                                                                 |                                                                           |
| `block-channels`       | channels                                                                                                |                                                                           |
| `bottleneck-reduction` | 4                                                                                                       |                                                                           |
| `notes`                | deep ResNet bottleneck                                                                                  |                                                                           |
| `transforms`           | RandomRotation(degrees=45), RandomHorizontalFlip(p=0.5), Normalize(mean=train_mean, std=train_std)    |                                                                           |
