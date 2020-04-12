# NLP-Train-Batch
Project: 2020SP-COSC6336-Natural Language Processing \
Task: Training on Batches for Sorting Letter using LSTM(Encoder-Decoder)+Attention

- - -

# Library:
- Python 3.7.4
- Scikit-Learn 0.21.2
- IDE: PyCharm 2020.1

- - -

# Final Project Guidelines:

- As you can see, the models are quite good at the task at hand. However, they are not perfect and you will see that the performance on the test data drops. One of the reasons is that the models were trained for a few epochs. Even though training for longer could improve the models, the code is not efficient because it does not use batches. Instead, we are iterating sample by sample without taking advantage of parallelization. Once you optimize this code, you will see that both models are good, and the attention brings a substantial boost in performance.
- Your task is to make this model efficient using batches. You can keep the same overall architecture, which is a single LSTM layer with one direction on both the encoder and decoder while also using attention. As requirement, you must use RNN-based models with attention, but the number of layers, directions, and hidden units of the RNN are up to you. Similarly, the attention method (either any version of Luong's attention or Bahdanau's attention) can be defined by you. You can also use self-attention if you prefer, but that is not required nor needed to achieve good results. More specifically, your notebook must include the following (gradable) aspects:
  - the notebook must show all the epoch iterations with their respective losses.
  - the resulting model must be selected based on the best performance on the validation set.
  - the notebook must display the training and validation losses in a single plot (x-axis being the epochs, and y-axis being the cross-entropy loss).
  - the notebook must display the results on the test set of your best model according to validation set.
  - the notebook must include a detailed description of how you are implementing batches, not only the code.
  - the notebook must specify in detail the architecture of your model.
  - the notebook must include an analysis section at the end where you detail a comparison of two or more different models that at least differ from using vrs. not using attention mechanism.
  - the notebook must be runnable by us, and it should skip training (**hint**: check if the `model.pt` file exists to skip training). We won't run the analysis section since that would require your other models to be included in the submission.
- We will measure the performance of your model using accuracy for the exact matches. Since this is a toy dataset, it's reasonable to measure the performance using this metric (it is worth mentioning that for tasks such as machine trasnlation, the official metric is BLEU). We will provide a test dataset (although auto-generated too) that is fixed for everyone to make all the systems comparable.

Instruction: https://github.com/mdrafiqulrabin/nlp-train-batch/blob/master/instruction.txt

- - -

# Result and Analysis:

We studied the 'Training on Batches for Sorting Letter using LSTM(Encoder-Decoder)+Attention' in this project. We *adopted* the demo code of [seq2seq](https://github.com/gaguilar/basic_nlp_tutorial/tree/master/tutorial_on_seq2seq_models) model and implemented the batch support for encoder and decoder. See below for summary:

Batch-Attention-Epoch: \
<img src="https://github.com/mdrafiqulrabin/nlp-train-batch/blob/master/result/summary/batch_attention_epoch.png" alt="batch_attention_epoch" width="600"/>

More complex dataset: \
<img src="https://github.com/mdrafiqulrabin/nlp-train-batch/blob/master/result/summary/complex_dataset.png" alt="complex_dataset" width="600"/>

Different prediction sizes: \
<img src="https://github.com/mdrafiqulrabin/nlp-train-batch/blob/master/result/summary/prediction_size.png" alt="prediction_size" width="600"/>

Check: https://github.com/mdrafiqulrabin/nlp-train-batch/blob/master/model.ipynb

- - -

# References:

- https://pytorch.org/docs/stable/nn.html 
- https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html 
- https://github.com/gaguilar/basic_nlp_tutorial/tree/master/tutorial_on_seq2seq_models 
- https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html 

