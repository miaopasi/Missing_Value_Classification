__author__ = 'Xiaolong Shen @ Nexd Tech'
"""
Reference: Krishan. P SmoothNN
Core Algorithm is R. Tibshirani GAM method for data preprocessing.
Several Part should be solved;
"""
'''
Issues in Github:
#5      Tidy Up OffLine Localization Project

#4		Merge the Project To Aeon Project.

#3		Implement Smooth Method.

#2		Implement Data Preprocessing

#1		Determine Which Method To Attempt
'''
import sys
from numpy import *
sys.path.append('./Ref_Code')
from loadfile import *


class Grid:
	def __init__(self, wifi_vec=None, core_pos=None, divergence=None):
		self.wifi_vec = []
		self.wifi = None
		self.core_pos = [];
		self.pos = None;
		self.divergence = None
		self.index = None

	def _set(self, wifi, pos, index):
		self.wifi = wifi;
		self.pos = pos;
		self.index = index;

	def _add(self, wifi_vec, core_pos):
		self.wifi_vec.append(wifi_vec)
		self.core_pos.append(core_pos)

	def _set_index(self, index):
		self.index = index


class Pos:
	def __init__(self):
		self.x = 0.0;
		self.y = 0.0;


class MeshGrid:
	def __init__(self):
		self.X_Min = None;
		self.X_Max = None;
		self.Y_Min = None;
		self.Y_Max = None;
		self.wifi_list = None;
		self.grids = {}; # Index: Class Grid

	def _set_param(self, xmi, xma, ymi, yma):
		self.X_Min = xmi;
		self.X_Max = xma;
		self.Y_Min = ymi;
		self.Y_Max = yma;

	def _add_grid(self, ind, grid):
		if ind in self.grids:
			print "[Error]Replicate Result"
		self.grids[ind] = grid

	def _set_wifilist(self, wifi_list):
		self.wifi_list = wifi_list;

class MeshData:
	def __init__(self):
		self.data = None
		self.mesh_data = MeshGrid()

	def _load(self, data):
		self.data = data

	def _get_index(self, pos):
		ind_x = floor(pos[0]) - self.X_Min
		ind_y = floor(pos[1]) - self.Y_Min
		ind = ravel_multi_index([int(ind_x),int(ind_y)],(int(self.X_Max-self.X_Min),int(self.Y_Max-self.Y_Min)))
		return ind

	def _grid_gen(self, wifi_vec, pos, ind):
		if ind not in self.mesh_data.grids:
			self.mesh_data._add_grid(ind, Grid())
		self.mesh_data.grids[ind]._set_index(ind)
		self.mesh_data.grids[ind]._add(wifi_vec, pos)

	def _mesh(self, window_size=1.0):
		"""
		Gonna Use Window Wrapping Method
		:param data: has component, wifi_matrix, wifi_pos, wifi_list
		:return: self.mesh_data
		"""
		self.X_Max = floor(self.data.wp_pos[:, 0].max()) + 1.
		self.Y_Max = floor(self.data.wp_pos[:, 1].max()) + 1.
		self.X_Min = floor(self.data.wp_pos[:, 0].min())
		self.Y_Min = floor(self.data.wp_pos[:, 1].min())
		self.mesh_data._set_param(self.X_Min, self.X_Max, self.Y_Min, self.Y_Max)
		self.mesh_data._set_wifilist(self.data.wifi_list)
		for i, pos in enumerate(self.data.wp_pos):
			ind = self._get_index(pos)
			wifi_vec = self.data.wifi_matrix[i, :]
			self._grid_gen(wifi_vec, pos, ind)



	def _normalization(self):
		# self.data[self.data != 0.] += 100.;
		self.data.wifi_matrix[self.data != 0] += 100.

	def _get(self, data):
		self._load(data)
		self._normalization()
		self._mesh()
		return self.mesh_data

	def get(self, data):
		return self._get(data)

	def get_by_path(self, wp_path, wifi_path):
		loader = LoadWifiData()
		train = loader.extract(wp_path, wifi_path);
		return self._get(train)


# au = AeonUtility();
loader = LoadWifiData()
train = loader.extract('./Data/Training/data_new.wp', './Data/Training/data_new.wifi')

mesh = MeshData()
mesh_data = mesh.get(train)
