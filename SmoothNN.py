__author__ = 'Xiaolong Shen'

from MeshGrid import MeshData
from numpy import *
import sys
sys.path.append('./Ref_Code')
from loadfile import *
from Classification_Base import *


class SNN(Classification_Base):
	def __init__(self):
		Classification_Base.__init__();

	'''
	Classification Part:
		Implement K-NN Classifier
	'''
	def _vec_format(self, vec):
		if vec.shape[0] != 1:
			try:
				vec = vec.reshape(1,len(vec))
			except Exception,e:
				print e
				return None
		return vec


	def _pearson(self, vec_a, vec_b):
		N = len(vec_a);
		up = (vec_a*vec_b).sum() - vec_a.sum()*vec_b.sum()/N
		down = sqrt((square(vec_a).sum() - (vec_a.sum())**2 / N)*(square(vec_b).sum() - (vec_b.sum())**2 / N))
		return up/down


	def _classifier(self, test_vec):

		test_ind = (test_vec != 0.)
		sim_vec = [];
		for i in range(self.NN_data.mat.shape[0]):
			train_vec = self.NN_data.mat[i,:]
			train_ind = (train_vec != 0.)
			ind = test_ind * train_ind
			if sum(ind) <= 5:
				sim_vec.append(-1.)
				continue
			res = self._pearson(train_vec[ind],test_vec[ind])
			sim_vec.append(-1. if isnan(res) else res)
		sim_vec = array(sim_vec)
		# print sim_vec.shape
		# print self.NN_data.mat.shape
		return sim_vec


	def clf_pos(self, test_vec, NN = 3):
		res = self._classifier(test_vec)
		res[isnan(res)] = -10;
		res_ind = argsort(res)[::-1]
		count = 0;
		pos = array([0.0, 0.0]);
		for ind in res_ind[0:NN]:
			if res[ind] < 0.5:
				return None
			count += res[ind]
			pos[0] += self.NN_data.pos[ind,0] * res[ind];
			pos[1] += self.NN_data.pos[ind,1] * res[ind];
		pos /= count;
		return pos

	def _validate_res(self, test_vec, test_tar):
		clf_pos = self.clf_pos(test_vec)
		if clf_pos is None:
			return 0.
		else:
			diff_pos = clf_pos - test_tar;
			D = sqrt(square(diff_pos).sum())
			return D



	def clf(self, test_vec):
		res = self._classifier(test_vec)
		return argmax(res)

	def test(self, test_data):
		res = []
		for test_vec in test_data:
			# t_res =

			pass;

	def validate(self, test_data, test_tar):
		RMSE = [];
		for i in range(test_data.shape[0]):
			test_vec = test_data[i,:]
			test_pos = test_tar[i,:]
			if len(test_vec[test_vec!=0.]) > 3:
				RMSE.append(self._validate_res(test_vec, test_pos))
			print "> %s / %s Finished" %(i, test_data.shape[0])
		print RMSE
		return mean(RMSE)

# clf = SNN()
# # clf.train_save('./Data/Training/data_new.wp', './Data/Training/data_new.wifi')
# clf.load_train()
#
# print clf.clf_pos(clf.NN_data.mat[1, :])
#
# print clf._validate_res(clf.NN_data.mat[1, :], clf.NN_data.pos[1,:])
# # print clf.NN_data.wifi_list

# print clf.validate(l.wifi_matrix, l.wp_pos)