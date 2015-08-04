__author__ = 'Xiaolong Shen @ Nexd Tech'

from SmoothNN import *
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

test_wifi_path = './Data/Training/data.wifi'
test_wp_path = './Data/Training/data.wp'

class SVM_CLF(Classification_Base):
	def __init__(self):
		Classification_Base.__init__(self)
		self.clf = svm.SVC()
		self.trained = False

	def _train(self):
		self.clf.fit(self.NN_data.mat, arange(self.NN_data.mat.shape[0]))
		self.trained = True;

	def _test(self, test_data):
		if self.trained:
			return self.clf.predict(test_data)
		else:
			print "[Error] Not Trained With Data, Empty Classifier"
			return None

	def _classifier(self, test_data):
		return self._test(test_data)


class NB_CLF(Classification_Base):
	def __init__(self):
		Classification_Base.__init__(self)
		self.clf = GaussianNB()
		self.load_train()
		self.trained = False
		self._train()

	def _train(self):
		self.clf.fit(self.NN_data.mat, arange(self.NN_data.mat.shape[0]))
		self.trained = True

	def _classifier(self, test_data):
		if self.trained:
			return self.clf.predict(test_data)
		else:
			print "[Error] Not Trained."
			return None

	def _proba_test(self, test_data, test_pos):
		res = self.clf.predict_proba(test_data)
		print res
		RMSE = []
		for i,proba in enumerate(res):
			# print "%s\n Var:%s Max:%s" %(proba, var(proba), max(proba))
			if var(proba) < 0.1 and max(proba) < 0.5:
				continue;
			ind = argsort(proba)[::-1]
			p = array([0., 0.])
			for j in range(5):
				p += self.NN_data.pos[ind[j]] * proba[ind[j]]
			p /= proba[ind[0:5]].sum()

			test_tar = test_pos[i,:]
			print "%s, %s" %(test_tar, p)
			RMSE.append(sqrt(square(test_tar - p).sum()))

		print RMSE
		print "VAR: %s" % var(RMSE)
		print "MEAN: %s" % mean(RMSE)
		print "MAX: %s" % max(RMSE)
		print "MIN: %s" % min(RMSE)

nb = NB_CLF()
nb._train()
# nb.validate_train()

# nb._proba_test(nb.NN_data.mat, nb.NN_data.pos)
print "========================================"
l = LoadWifiData()
l.extract_with_ref(test_wp_path, test_wifi_path, list(nb.NN_data.wifi_list))
nb._proba_test(l.wifi_matrix, l.wp_pos)
# nb.validate(test_wp_path, test_wifi_path)