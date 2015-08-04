__author__ = 'Xiaolong Shen @ Nexd Tech'

from numpy import *
import sys
sys.path.append('./Ref_Code')
from loadfile import *
from MeshGrid import MeshData

class GridNN:
	def __init__(self):
		self.mat = None
		self.pos = None
		self.wifi_list = None

	def _set(self, mat, pos):
		self.mat = mat;
		self.pos = pos;

	def _set_wifilist(self, wifi_list):
		self.wifi_list = list(wifi_list)


class Classification_Base:
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

	def _classifier(self, test_data):
		res = array([]);
		"""
		For Other Application. Just replace the Classifier function will do
		"""
		return res

	def _validate(self, test_data, test_pos):
		res = self._classifier(test_data)
		RMSE = []
		for i,ind in enumerate(res):
			test_tar = test_pos[i, :]
			train_tar = self.NN_data.pos[ind, :]
			RMSE.append(sqrt(square(test_tar - train_tar).sum()));
		print RMSE
		print "VAR: %s" % var(RMSE)
		print "MEAN: %s" % mean(RMSE)
		print "MAX: %s" % max(RMSE)
		print "MIN: %s" % min(RMSE)
		# return mean(RMSE)

	def validate(self, test_wp_path, test_wifi_path):
		l = LoadWifiData();
		l.extract_with_ref(test_wp_path, test_wifi_path, list(self.NN_data.wifi_list))
		return self._validate(l.wifi_matrix, l.wp_pos);

	def validate_train(self):
		return self._validate(self.NN_data.mat, self.NN_data.pos)
