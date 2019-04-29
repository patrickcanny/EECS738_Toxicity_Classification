# EECS738_Toxicity_Classification
Toxicity Classification and Mitigation. See: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

## Background
We really liked the ideas behind the Hidden Markov Model Project, so we decided to try and tackle a project centered around similar principles. We saw this cool competitioon on Kaggle and decided to use the dataset to try and create a ML approach. For more detailed information on all aspects of this project, please see our Jupyter notebooks and our final presentation.
## Approach
We ultimately decided to use a word-toxicity weighting method, combined with a simple Keras neural network, to accomplish this task. For detailed information on this approach, see the 'Comment Predictor' jupyter notebook. For detailed information on another approach which uses the highly-successful LSTM method for classifying toxicity, see the 'LSTM Comment Predictor' notebook.
## Findings
Our word weighting method reaches ~70% accuracy on the training set, and provides some distinction in prediction results between comments classified as toxic and non-toxic. The LSTM method is capable of reaching >85% accuracy and providing better prediction results.
## Notes
Please ensure that you install Git LFS within your local repository to ensure that the large .csv data files are pulled correctly. If you cannot retrieve the dataset this way, it can also be downloaded from the 'Unintended Bias in Toxicity Classification' Kaggle competition page and should be placed in the /data subfolder.

## References
### Kaggle competitions
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
### LSTM Method
https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras