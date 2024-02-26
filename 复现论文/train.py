from  simpletransformers.classification import ClassificationModel,ClassificationArgs
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
cuda_available = torch.cuda.is_available()


train = pd.read_csv('./content/my_train.csv')
train = train.dropna()
train1 = train
train = train[['sentence1','sentence2','label']]
train.columns = ["text_a", "text_b", "labels"]
train1 = train1[['sentence1','sentence2','label']]
train1.columns = ["text_a", "text_b", "labels"]

test = pd.read_csv('./content/my_test.csv')
# test = test.dropna()
test = test[['sentence1','sentence2']]
test.columns = ["text_a", "text_b"]

args = ClassificationArgs()
args.num_train_epochs = 20
args.learning_rate = 2e-5
args.overwrite_output_dir =True
args.train_batch_size = 128
args.eval_batch_size = 128
args.use_cached_eval_features = False
args.use_multiprocessing = False
args.reprocess_input_data = True

label = len(set(train['labels']))

label=2
model = ClassificationModel('auto', 'xlm-roberta-base', num_labels=label,use_cuda = cuda_available,args=args)



from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base')

label=2
model = ClassificationModel('auto', 'xlm-roberta-base', num_labels=label,use_cuda = cuda_available,args=args)

# Train the model
model.train_model(train1)

# Evaluate the model on the dev set
# result, model_outputs, wrong_predictions = model.eval_model(test_data)
# output = [np.argmax(i,axis = 0) for i in model_outputs ]
result, model_outputs, wrong_predictions = model.eval_model(train[:120])
output = [np.argmax(i,axis = 0) for i in model_outputs ]
print('accuracy',accuracy_score(train[:120]['labels'],output))
print('weighted',f1_score(train[:120]['labels'],output,average = 'weighted'))
print('macro',f1_score(train[:120]['labels'],output,average = 'macro'))

result, model_outputs, wrong_predictions = model.eval_model(train[10000:10000+150])
output = [np.argmax(i,axis = 0) for i in model_outputs ]
print('accuracy',accuracy_score(train[10000:10000+150]['labels'],output))
print('weighted',f1_score(train[10000:10000+150]['labels'],output,average = 'weighted'))
print('macro',f1_score(train[10000:10000+150]['labels'],output,average = 'macro'))

to_predict = []
for i in range(len(test['text_a'])):
  u1 = []
  if str(test['text_a'][i]) == 'nan':
      u1.append('a')
  else:
      u1.append(test['text_a'][i])

  if str(test['text_b'][i]) == 'nan':
      u1.append('b')
  else:
      u1.append(test['text_b'][i])

  to_predict.append(u1)

  predictions, raw_outputs = model.predict(to_predict)

  test = pd.read_csv('./content/my_test.csv')
  # test = test.dropna()
  # test = test[['sentence1','sentence2']]
  # test.columns = ["text_a", "text_b"]
  test = test['id']
  test['label'] = predictions
  test.to_csv('submission.csv', index=False)