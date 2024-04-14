# HW3P2: Submission Details

## Running Intructions:
The Jupyter notebook should be self contained, and the cells can be ran as is. *HW3P2_Alec.ipynb* directly took cells from HW3P2 Starter Notebook.

## Experiments:

**Resources:**
1. https://wandb.ai/atrela-cmu/hw3p2-ablations?nw=nwuseratrela
    
For this assignment I started off by filling our the pBLSTM skeleton code. This took a good two days of implementation in the dataloader and pyramidal bidirectional-LSTM block. Once I was able to run the code for the first round of training I was getting extremely high errors, on the order of hundreds. This ended up being due to the fact that all label strings were of a fixed length. I needed to remove the extra empty space from the label strings to match the label strings. 

After doing this, I was able to hit the high-cutoff on my first attempt. I had a simple arrangement of a couple convolutional layers in the encoder with sufficient padding so as not to reduce the dimensionality and representation in the embedding. I also had two pBLSTM blocks. In the decoder, I started with a couple of linear layers with batchnorm. This got me a Levenschtein distance under 6, so I had already hit the high cutoff. It seemed like I overfit the training data, so I attempted to get some marginal improvement by adding some dropout to the embedding and pBLSTM as well as the decoder. Additionally, I added some time and frequency masking to the input data to add some variability to the training regimine. This caused some non-negligible improvement to performance, letting me drom from a distance of ~5.8 to ~5.2 on the training set. I didn't need too many runs to hit these metrics. My intuitions from assignment 1 translated well to this exercise. 

## Highest Performing Model:

| Parameter   | Value                                        | Description                                 |
|-------------|----------------------------------------------|---------------------------------------------|
| lr          | 0.0015                                       | Learning rate for the training process      |
| gamma       | 0.4                                          | Decay factor for the learning rate          |
| model       | benchmark-more-MORE-pBLSTMs-transf           | The name of the model configuration         |
| epochs      | 50                                           | Total number of training epochs             |
| time_p      | 0.3321228618686698                           | Probability of time masking occurring       |
| freq_mask   | 27                                           | Frequency masking parameter                 |
| time_mask   | 149                                          | Time masking parameter                      |
| batch_size  | 128                                          | Batch size (increase if your device allows) |
| beam_width  | 10                                           | The beam width used in the model            |
| embed_size  | 220                                          | Size of the embedding layer                 |
| milestones  | 15, 20, 25, 30, 35, 40, 45                   | Epoch milestones for updating learning rate |
