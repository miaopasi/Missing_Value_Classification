__author__ = 'Xiaolong Shen'

from MeshGrid import MeshData
from numpy import *
import sys
sys.path.append('./Ref_Code')
from loadfile import *


class GridNN:
	def __init__(self):
		self.mat = None
		self.pos = None
		self.wifi_list = None

	def _set(self, mat, pos):
		self.mat = mat;
		self.pos = pos;

	def _set_wifilist(self, wifi_list):
		self.wifi_list = wifi_list;


class SNN:
	def __init__(self):
		self.mesh = MeshData()
		self.mesh_data = None;
		self.NN_data = GridNN();

	'''
	Data Processing Part:
		Including Smoothing and Grid-lize
	'''
	def _grid_smooth(self, grid):
		pos = asarray(grid.core_pos).mean(axis=0)
		vec = zeros(grid.wifi_vec[0].shape)
		count = zeros(grid.wifi_vec[0].shape)
		for v in grid.wifi_vec:
			vec[v!=0.] += v[v!=0.]
			count[v!=0.] += 1.;
		vec[count != 0.] /= count[count!=0.]
		return vec, pos

	def _tidy_mesh(self):
		L = len(self.mesh_data.grids.keys())
		W = len(self.mesh_data.wifi_list)
		mat = zeros((L, W))
		pos = zeros((L, 2))
		for i, key in enumerate(self.mesh_data.grids.keys()):
			v, p = self._grid_smooth(self.mesh_data.grids[key])
			mat[i,:] = v;
			pos[i,:] = p
		return mat, pos

	def _get_NN(self):
		m, p = self._tidy_mesh()
		self.NN_data._set(m, p)
		self.NN_data._set_wifilist(self.mesh_data.wifi_list)

	def train(self, wp_path, wifi_path):
		print "> Start Training"
		self.mesh_data = self.mesh.get_by_path(wp_path, wifi_path);
		print "> Loaded Data"
		self._get_NN()
		print "> Training Done"

	def train_save(self, wp_path, wifi_path, save_path='SmoothNN.npz'):
		self.train(wp_path, wifi_path)
		savez(save_path, mat = self.NN_data.mat, pos = self.NN_data.pos, wifi_list = self.mesh_data.wifi_list)

	def load_train(self, load_path='SmoothNN.npz'):
		data = load(load_path)
		self.NN_data._set(data['mat'],data['pos'])
		self.NN_data._set_wifilist(data['wifi_list'])

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

	# def _classifier(self, test_vec):
	#
	# 	mat = self.NN_data.mat[:, test_vec!=0.]
	# 	t_vec = test_vec[test_vec!=0.]
	# 	test_vec = self._vec_format(t_vec)
	# 	if test_vec is None:
	# 		return None
	# 	# print test_vec
	# 	# print mat
	#
	#
	# 	L = mat.shape[0]
	# 	N = mat.shape[1]
	# 	# /sum{x*y}
	# 	p1 = mat.dot(test_vec.T).T[0]
	# 	# /sum{x} * /sum{y} / N
	# 	p2 = mat.mean(axis = 1) * test_vec.sum()
	# 	# /sum{x^2} - (/sum{x})^2/N
	# 	p3 = square(mat).sum(axis = 1) - (mat.sum(axis=1))**2/N
	# 	# /sum{y^2} - (/sum{y})^2/N
	# 	p4 = square(test_vec).sum() - (test_vec.sum())**2/N
	# 	sim_vec = (p1 - p2) / sqrt(p3*p4)
	# 	del p1,p2,p3,p4
	# 	# print sim_vec
	# 	# print "==================================================="
	# 	return sim_vec

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

clf = SNN()
# clf.train_save('./Data/Training/data_new.wp', './Data/Training/data_new.wifi')
clf.load_train()

print clf.clf_pos(clf.NN_data.mat[1, :])

print clf._validate_res(clf.NN_data.mat[1, :], clf.NN_data.pos[1,:])
# print clf.NN_data.wifi_list
l = LoadWifiData()
l.extract_with_ref('./Data/Training/data_new.wp', './Data/Training/data_new.wifi', list(clf.NN_data.wifi_list))
print clf.validate(l.wifi_matrix, l.wp_pos)