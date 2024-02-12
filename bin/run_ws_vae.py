#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import sys
import os
path_to_package = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(os.path.join(path_to_package, 'src'))
from var_logger import load_wrench_data
from ws_vae import WeakSupervisionVAE


#### Load dataset
dataset_name = 'census'
# train split
dataset_path = os.path.join(path_to_package, 'data', dataset_name, 'train.json')
train_data = load_wrench_data(dataset_path)
# test split
dataset_path = os.path.join(path_to_package, 'data', dataset_name, 'test.json')
test_data = load_wrench_data(dataset_path)


#### Train WS-VAE
model = WeakSupervisionVAE(train_data)
model.fit(dataset_train=train_data, gamma=0.1)

#### Test WS-VAE
f1 = model.test(test_data, 'f1_binary')
accuracy = model.test(test_data, 'acc')
auc = model.test(test_data, 'auc')
print('f1-score: {}'.format(f1))
print('accuracy: {}'.format(accuracy))
print('ROC-AUC score: {}'.format(auc))