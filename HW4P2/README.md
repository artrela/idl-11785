# HW4P2: Submission Details

## Running Intructions:
The Jupyter notebook should be self contained, and the cells can be ran as is. *HW4P2_Alec.ipynb* directly took cells from HW4P2 Starter Notebook.

## Experiments:

**Resources:**
1. https://wandb.ai/atrela-cmu/HW4P2-S24/workspace?nw=nwuseratrela 
    
My work flow essentially consisted of filling in all of the blanks given in the starter code. I was busy with the project, so I used all of my slack days for this assignmnet. This allowed me to benefit from the pretrained weights given at a later points and all of the notebook changes made during the course of this assignment. I started by getting my implementation to train the decoder on the train-clean-50 dataset. Once I saw that this was working, I switched to the train-clean-100 dataset and simply ran the code as expected, retraining the decoder and then eventually the encoder. I did not have to change any of the given hyperparameters besides batch size and the decoder layers which I changed to 16, 3, and 3 respectively. 

There was little need for me to tune hyperparameters on this assignment as the blanks to fill in largely sufficed on the first fun. 

## Highest Performing Model:

| Category             | Parameter            | Value                                         | Notes                                                                 |
|----------------------|----------------------|-----------------------------------------------|-----------------------------------------------------------------------|
| General              | run_name             | "baseline"                                    |                                                                       |
| General              | Name                 | "Alec"                                        | write your name here for study group                                 |
| Dataset              | train_dataset        | "train-clean-100"                             | train-clean-50 (primarily for debugging purposes), train-clean-100    |
| Dataset              | cepstral_norm        | True                                          |                                                                       |
| Dataset              | input_dim            | 27                                            |                                                                       |
| Dataset              | batch_size           | 16                                            | 64 (decrease this as you modify the network architecture)             |
| Encoder              | enc_dropout          | 0.2                                           | [0.1, 0.4]                                                            |
| Encoder              | enc_num_layers       | 1                                             | [1, 3]                                                                |
| Encoder              | enc_num_heads        | 1                                             | [1, 4]                                                                |
| Decoder              | dec_dropout          | 0.2                                           | [0.1, 0.4]                                                            |
| Decoder              | dec_num_layers       | 3                                             | [1, 3]                                                                |
| Decoder              | dec_num_heads        | 3                                             | [1, 4]                                                                |
| Network Parameters   | d_model              | 512                                           | [256, 1024]                                                           |
| Network Parameters   | d_ff                 | 2048                                          | [512, 4096]                                                           |
| Learning Rate        | learning_rate        | 1E-4                                          | [1E-3, 1E-4], this will depend on the specified optimizer            |
| Optimizer            | optimizer            | "AdamW"                                       | Adam, AdamW                                                           |
| Optimizer            | momentum             | 0.0                                           | if SGD                                                                |
| Optimizer            | nesterov             | True                                          | if SGD                                                                |
| Scheduler            | scheduler            | "CosineAnnealing"                             | CosineAnnealing, ReduceLR                                             |
| Scheduler            | factor               | 0.9                                           | if ReduceLR                                                           |
| Scheduler            | patience             | 5                                             | if ReduceLR (note on scheduling and patience relation)                |
| Training Parameters  | epochs               | 50                                            |                                                                       |
