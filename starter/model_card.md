# Model Card

## Model Details
- Multi-layer perceptron with relu activation function and using dropout to reduce overfitting. 
- The model is written using Pytorch. CrossEntropy loss is used to train the model. 
- Hyper-parameters selection is done using Optuna.

## Intended Use
- This model should be used to predict whether income of a US citizen exceeds $50K/yr based on census data. 
## Training Data
- The data was obtained from the [Census Bureau](https://archive.ics.uci.edu/ml/datasets/census+income). The target 
 class is the 'salary', it is a binary variable which value is in '<=50k' and '>50k'. 
- The raw data need to be processed by removing spaces.
- The original data set has 32561 rows and 15 columns. An 80-20 split was used to break this into a train and test set. 
No stratification was done. Within the train set another 80-20 split is done to split data between training and 
validation so that properly evaluate the performance of hyper-parameters and choose the optimal level for these 
parameters. Once the hyper-parameters have been set we use all the training data for calibrating the model. 
- To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels. 
A scaler was also used on continuous variables.

## Evaluation Data
- As described in the previous section, the evaluation data represent 20% of the original data. The label binarizer, 
the One Hot Encoder and the  scaler used for training data are used so that we can proceed with inference on test data.

## Metrics
- The model was evaluated using precision, recall and F1 score. 
- The values obtained on test data are: precision: 0.78, recall: 0.49, F1: 0.60.

## Ethical Considerations
This data is collected from a survey, there could be bias coming from how the survey was set-up or on how people 
properly answer to the different questions corresponding to the features used for training the model.

## Caveats and Recommendations
More work should be done on hyper-parameter tuning to improve the performance of the model. This project is about CI/CD,
so more emphasis was put on building an architecture using GitHub Action, DVC+S3, FastAPI and Heroku than in 
fine-tuning the model.
