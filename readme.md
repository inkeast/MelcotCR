# Supplementary Material
This replication package contains supplementary material for the paper "meCoTCR: Improving Code Review Via Maximum-Entropy Chain-of-Thought Reasoning". The package is organized as follows:

* `./Data` The datasets used in this paper
  * `./Data/TrainDataset.parquet` The MEFT training dataset used in the paper.
  * `./Data/TestDataset.parquet` The test dataset used in the paper.
* `./Trainer` The fine-tuning code for the review comment generation task.
* `./Evaluation` The code related to evaluation includes generating review comments, calculating IoU scores, and using LLM-as-Judge to assess the accuracy of review comments.