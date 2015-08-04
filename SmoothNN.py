__author__ = 'Xiaolong Shen'

from MeshGrid import MeshData

class GridNN:
	def __init__(self):
		self.mat = None
		self.pos = None


class SNN:
	def __init__(self):
		self.mesh_data = MeshData();

	def train(self, wifi_path, wp_path):
		self.mesh_data.get_by_path(wifi_path, wp_path);


