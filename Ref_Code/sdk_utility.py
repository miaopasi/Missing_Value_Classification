# coding: UTF-8
__author__ = 'Xiaolong Shen @ Nexd Tech'

'''
THIS IS THE UTILITY TOOLKIT FOR SDK GENERATED DATA.

TEMPORARILY DEVELOPED FOR AEON USAGE.

ADD-ON FUNCTION NEED TO BE FILLED.
'''
from numpy import *


class ClfDataStorage:
	def __init__(self):
		self.wifi_matrix = None
		self.miss_mac_count = None
		self.hit_mac_count = None
		self.total_mac_count = None
		self.user = None
		self.timestamp = None

	def set(self, wifi_matrix, miss_mac_count, hit_mac_count, total_mac_count, user, timestamp):
		self.wifi_matrix = wifi_matrix
		self.miss_mac_count = miss_mac_count
		self.hit_mac_count = hit_mac_count
		self.total_mac_count = total_mac_count
		self.user = user
		self.timestamp = timestamp

class SDKUtility:
	def __init__(self):
		self.drift = 120;
		pass
	"""
	PUBLIC FUNCTION:
		EXTRACT_WIFI    EXTRACTING WIFI FILE ONLY ACCORDING TO REFERENCE LIST FOR CLASSIFICATION USAGE
	"""
	def extract_wifi(self, wifi_path, ref_list=None):
		"""
		:param wifi_path:
		:param ref_list:
		:return:
			miss_count
			hit_count
			wifi_matrix
		"""
		if ref_list is not None:
			user, ts, wifi_vec, tc, mc, hc = self._extract_wifi(wifi_path, ref_list)
			cd = []
			for i,n in enumerate(user):
				tt = ClfDataStorage()
				tt.set(array([wifi_vec[i,:]]), mc[i], hc[i], tc[i], user[i], ts[i])
				cd.append(tt)
			return cd
		else:
			print "No Reference List Found"
			return None


	"""
	PRIVATE FUNCTION
	"""

	def _get_task_number(self, wifi_path):
		f = open(wifi_path)
		L = f.readlines()
		f.close()
		count = 0;
		taskid = ''
		tasklist = []
		for i,l in enumerate(L):
			ls = l.split('#')
			task = ls[1]
			if task == taskid:
				pass
			else:
				taskid = str(task)
				count += 1
				tasklist.append(taskid)
		return count, tasklist

	def _extract_wifi(self, wifi_path, ref_list):
		"""
		Data Format:
			用户 ID#采集 ID#SSID#BSSID#Capability#Level#Frequency#扫描时间#强度（以100为准）#写入时间（毫秒

		:param wifi_path: path of wifi file(filepath)
		:param ref_list: reference wifi list
		:return:
			user:   USER UNIQUE ID
			ts:     TIMESTAMP OF THIS RECORD
			wifimatrix:     WIFI_VECTOR FOR CLASSIFICATION USAGE
		"""
		f = open(wifi_path)
		lines = f.readlines()
		f.close()
		tn, tl = self._get_task_number(wifi_path)
		# print "Task Count: %s, list: %s" % (tn, tl)
		wifimatrix = zeros((tn, len(ref_list)))
		total_count = 0
		miss_count = 0
		hit_count = 0
		user = [0]*tn
		ts = [0]*tn
		total_count = [0]*tn
		miss_count = [0]*tn
		hit_count = [0]*tn
		for i, l in enumerate(lines):
			ls = l.split('#')
			if len(ls) is not 10:
				continue
			mac = ls[3]
			task_id = ls[1]
			ss = int(ls[5]) + self.drift;
			task_ind = tl.index(task_id)
			user[task_ind] = ls[0]
			ts[task_ind] = int(ls[9])
			try:
				insert_pos = list(ref_list).index(mac)
				wifimatrix[task_ind, insert_pos] = ss
				hit_count[task_ind] += 1
			except Exception, e:
				# print e
				miss_count[task_ind] += 1
			total_count[task_ind] += 1

		return user, ts, wifimatrix, total_count, miss_count, hit_count

