'''
===================================================================
Project Name    : Differential Evolution
File Name       : configuration.py
Encoding        : UTF-8
Creation Date   : 2021/10/19
===================================================================
'''

import numpy as np

class Configuration:
	def __init__(self):
		#入出力設定
		self.path_out   = './'
		self.log_name   = '_result_' + 'DE'
		self.log_out    = True

		# DEの設定
		self.max_pop    = 50           # 個体数
		#self.max_gen    = 600         # 最大世代数(今回はmax_evalsで制限)
		self.F          = 0.5          # スケール因子
		self.CR         = 0.9          # 交叉率

		# 問題設定
		self.prob_dim   = 20            # 問題の次元数
		self.prob_name  = ['F1','F5']   # 解く問題

		# 実験環境
		self.max_trial  = 30            # 試行回数
		self.max_evals  = 100000        # 評価回数(max_pop × max_gen)

	# ランダムシード値設定
	def setRandomSeed(self, seed = 1):
		self.seed = seed
		self.rd = np.random
		self.rd.seed(self.seed)
