# HS_classification

## Introduction
(설명추가예정)

## Requirements
It has been tested under environment.

* python ==3.9.6
* numpy == 1.21.1
* pandas ==1.3.1
* pytorch == 19.0.9
* transformers == 4.9.2
* sklearn == 0.24.2
* tqdm == 4.62.0

## Step1. Train the model
Run `train.py` to train the model for the classification. It has the following arguments:

* `--epochs`: Number of training epochs (100 by default).
* `--batch_size`: Batch size (64 by default).
* `--num_labels`: Number of classes. When classifiying the subheadings, it is 925 (925 by default).
* `--model`: NLP model to use. It can be "kobert", "koelectra", "klue".
* `--load`: File name to continue the training (None by default).
* `--learning_rate`: Learning rate (1e-5 by default).
* `--data_path`: Path where the data for the train is stored.
* `--output_path`: Path to store the model.
```
$ python train.py --model koelectra
```
## Step2. Evaluate the model
Run `evaluation.py` to evaluate the model trained at step1. It has the following arguments:

* `--batch_size`: Batch size (64 by default).
* `--num_labels`: Number of classes. When classifiying the subheadings, it is 925 (default).
* `--model`: NLP model to use. It can be "kobert", "koelectra", "klue".
* `--model_path`: File name to load the trained classifition model in step1.
* `--data_path`: Path where the data for the evaluation is stored.
```
$ python evaluation.py --model koelectra --model-path ./output/model_85.pt
```
## Step3. Use the model for the prediction and supporting facts
Run `retrieval.py` to get the hsk code suggestion and supporting facts for the input description. It has the following arguments:

* `--batch_size`: Batch size (64 by default).
* `--num_labels`: Number of classes. When classifiying the subheadings, it is 925 (default).
* `--model`: NLP model to use. It can be "kobert", "koelectra", "klue".
* `--model_path1`: File name to load the trained classifition model in step1.
* `--model_path2`: File name to load the trained model for the embedding. 
* `--input_desc`: File name where input description is stored.
* `--highlight_num`: Number of sentences to highlight (7 by default).
* `--compete_num`: Number of subheadings to show (3 by default).
```
$ python evaluation.py --model koelectra --model-path1 ./output/model_85.pt --input_desc ./input.txt
```
## Expected Results
(설명추가예정)

|  | | HS4 | | | HS6 | | 
| --- | --- | --- | --- | ---  | ---  | --- | 
|  | k=1 | k=3 | k=5 | k=1  | k=3 | k=5 | 
| --- | --- | --- | --- | ---  | ---  | --- | 
| KoBERT | 84.51 | 91.07 | 92.22 | 78.88 | 87.70 | 90.06 | 
| KoELECTRA | 87.48 | 93.42 | 94.92 | 83.22 | 90.99 | 92.88 | 
| KLUE-RoBERTa | 89.24 | 95.50 | 96.49 | 86.06 | 93.98 | 95.25 | 
