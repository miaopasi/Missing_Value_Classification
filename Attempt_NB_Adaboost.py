"""
Attempt With NB+Adaboost Method. Failed as the Adaboosted NB is still a NB. The bias and the deterministic method might not fit the boosting method.
"""


import sys
sys.path.append("./Ref_Code")

from numpy import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from kernel import *

au = AeonUtility();
train = au.load_wifi('./Data/Training/data_new.wp', './Data/Training/data_new.wifi')
# savez('Aeon_Base_Data.npz', wifi_list=train.wifi_list, wp_pos=train.wp_pos, all_data=train)
test = au.load_wifi('./Data/Training/data.wp', './Data/Training/data.wifi', train.wifi_list)

print "> Loading Finished"
print train.wifi_matrix.min();
print train.wifi_matrix.max();

#
ak = AeonKernel();
ak.train_NB(train.wifi_matrix, arange(train.wifi_matrix.shape[0]))
print "> Training Finished"
# ak.save()
# print "> Saved"
# ak.load_clf()
print "> Load In"
import time

st = time.time()
ak.validate_test_accuracy_proba(test.wifi_matrix, test.wp_pos, train.wp_pos)
ed = time.time()
print "> Validation Done"
print "> Time Comsumption for Test : %s, Average Time Comsumption : %s" %(ed-st, float(ed-st)/test.wifi_matrix.shape[0])
