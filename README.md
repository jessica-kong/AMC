
In all the models, 'data_en/data_cn' should contain train/val/test set of one fold, and users should try 5-fold validation. Full datasets are given in dataset.zip.

※ BERT, Roberta, BART and T5: run the python file that ends with 'run.py' in each folder. 

※ GLM models: (1) run train.sh, (2) run evaluate.sh

※ convert_dataset_to_jsonl.py is used to convert txt in train/val/test folder to a json file. Hard prompt can be added in this step.

※ data_analysis.py is used to calculate accuracy and F1 scores. Inputs can be json, txt or xls.

※ Code in the package takes English dataset as the input. The main difference between dealing with Chinese dataset and English dataset is the pre-trained models. 

