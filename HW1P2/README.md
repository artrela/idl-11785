# HW1P2: Submission Details

## Running Intructions:
Both Jupyter notebooks should be self contained, and the cells can be ran as is. *hw1p2.ipynb* directly took cells from *HW1P2_S24_Starter_Notebook* with the added extension of generating a WandB sweep. 

## Experiments:

**Resources:**
1. https://docs.google.com/spreadsheets/d/1RS68V5n5cz4EbRjuQ3qtgpqolAyWjVdzAhjE8iE0U_M/edit#gid=0
2. https://wandb.ai/atrela-cmu/hw1p2?workspace=user-atrela
3. https://wandb.ai/atrela-cmu/hw1p2-sweep?workspace=user-atrela 

I began with an initial approach of using a diamond funnel, slowly increasing the depth and dropout rates to reduce overfitting. After reading the original BatchNorm paper, I additionally added those after each layer, with a added benefit of being able to increase the learning rate. Around this time I also switched to the AdamW optimizer after reading that it can afford a wider range in the basin of acceptable hyperparameters that would lead to an high-performant solution. All of these results and trends can be seen in [1, 2].   

At this point, I felt that I saturated my ability to manually optimize my hyperparameters, so I made a new notebook *hw1p2.ipynb*, that integrated the sweep capabilities of WandB. This enabled me to try a variable amount of learning rates, schedulers, activations, dropout rates, and context. I learned from my teammates that an increase of context leads to an increase in results. At this point, I had also been using one scheduler, but I felt that this lead to a decrease of performance once the learning rate became sufficiently small. Due to this, I learned that chaining two schedulers could help to resolve this issue. it allowed me to keep the learning rate sufficiently high until learning began to plateau. From this sweep, I learned that I could perform a sort of "learning-rate warmup" with a linear scheduler for N epochs before switching to a ReduceRLonPlateau scheme. After trying another sweep with these newfound results, I learned that I had been normalizing data across the incorrect axis. After resolving this, I was able to see my best performing model which obtained ~87% accuracy, jumping upwards of 3% from prior. These results are present in [3]. 

Somewhere in the testing scheme I also started incorperating probabilistic masking of the input data. With a 50% chance I would add frequency masking and with 50% chance I woudl add time masking. This meant that on expectation the data would be 25% unaltered data, 25% frequency masking, 25% time masking, and 25% both. 

## Highest Performing Model:

| Hyperparameters | Value                    |
|-----------------|--------------------------|
| Activation      | GeLu                     |
| Scheduler       | LinearLR(default_params) |
| Batch Size      | 2048                     |
| Context         | 35                       |
| Dropout         | 0.25                     |
| Epochs          | 40                       |
| Initial LR      | 0.009                    |
| Mask            | 6                        |