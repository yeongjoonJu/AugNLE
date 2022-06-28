## Self-Training

### file directory
*** annotation ***
- VQA-X/annotated/vqaX_train.json | vqaX_val.json
- nocaps/filtered_78.51_k-100_p-0.5_t-0.7_n_1.json
  
*** image ***
- image/train2014 | val2014
- image/nocaps

*** dataloader ***
- train_teacher | val_teacher : data preprocessing to train in teacher mode
- train_student | val_student : data preprocessing to train in student mode
- pseudo_teacher : data preprocessing to psuedo label in teacher mode
- pseudo_student : data preprocessing to psuedo label in student mode

*** Algorithm ***
in 10 iteration
1. preprocessing teacher mode dataset (VQA-X)
2. Training (1) dataset to NLX-GPT teacher mode
3. Inference to pseudo label captioning dataset
4. preprocessing student mode dataset (VQA-X and (3) pseudo labeled dataset)
5. Trainint (4) dataset to NLX-GPT student mode
6. Inference to pseudo label captioning dataset
7. Repeating (1) ~ (6)