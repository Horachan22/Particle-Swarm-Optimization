'''
===================================================================
Project Name    : Differential Evolution
File Name       : optimizer.py
Encoding        : UTF-8
Creation Date   : 2021/10/19
===================================================================
'''

import sys
import function as fc
import numpy as np
import copy

# 差分進化クラス
class DifferentialEvolution:
	def __init__(self, cnf, fnc):
		self.cnf = cnf                                                 # 設定
		self.fnc = fnc                                                 # 関数
		self.pop = []                                                  # 個体群

	# 次世代個体群生成
	def get_next_population(self):
		next_pop = []                                                  # 次世代個体群
		for i in range(len(self.pop)):                                 # 全ての個体に対して
			v_t = self.select_parent()                                 # 変異戦略
			u_t = self.cross_over(self.pop[i], v_t)                    # 交叉
			next_pop.append(self.compare_solution(self.pop[i], u_t))   # 比較
		self.pop = next_pop

	# STEP1: 初期化
	def initialize_solutions(self):
		for i in range(self.cnf.max_pop):                              #
			self.pop.append(Solution(self.cnf, self.fnc))              # 個体群に個体を追加
			self.get_fitness(self.pop[i])                              # 個体番号の割り当て

	# STEP2: 変異戦略(親個体の選択)
	def select_parent(self):
		v_t = Solution(self.cnf, self.fnc)                             # 変異個体用の変数の生成
		r = []                                                         # 重複のない乱数の生成
		while len(r) < 3:                                              # ↓
			n = self.cnf.rd.randint(0, len(self.pop))                  # ↓
			if not n in r:                                             # ↓
				r.append(n)                                            # ここまで
		p_1 = self.pop[r[0]]                                           # 親個体1
		p_2 = self.pop[r[1]]                                           # 親個体2
		p_3 = self.pop[r[2]]                                           # 親個体3
		v_t.x = p_1.x + self.cnf.F * (p_2.x - p_3.x)                   # 変異個体の生成
		return v_t

	# STEP3: 交叉(子個体の生成)
	def cross_over(self, x_t, v_t):
		u_t = Solution(self.cnf, self.fnc)                             #
		J_rand = self.cnf.rd.randint(0, self.cnf.prob_dim)             #
		for J in range(self.cnf.prob_dim):                             #
			if J == J_rand or self.cnf.rd.rand() <= self.cnf.CR:       #
				u_t.x[J] = v_t.x[J]                                    # 子個体の遺伝子に変異個体の遺伝子を格納
			else:                                                      #
				u_t.x[J] = x_t.x[J]                                    # 子個体の遺伝子に親個体の遺伝子を格納
		u_t.check_domain()                                             # 定義域確認
		return u_t

	# STEP4: 親個体と子個体の比較
	def compare_solution(self, x_t, u_t):
		self.get_fitness(u_t)                                          # 子個体の評価値を計算
		if x_t.f > u_t.f:                                              # 親個体vs子個体
			return u_t                                                 # ↓
		else:                                                          # ↓
			return x_t                                                 # ここまで

	# 評価値fの計算
	def get_fitness(self, solution):
		solution.f = self.fnc.doEvaluate(solution.x)                   #

#個体のクラス
class Solution:
	def __init__(self, cnf, fnc, parent = None):
		self.cnf, self.fnc, self.x, self.f = cnf, fnc, [], 0.
		# 個体の初期化
		if parent == None:
			self.x = [self.cnf.rd.uniform(self.fnc.axis_range[0], self.fnc.axis_range[1]) for i in range(self.cnf.prob_dim)]
		# 親個体のコピー
		else:
			self.x = [parent.x[i] for i in range(self.cnf.prob_dim)]
		# リスト -> ndarray
		self.x = np.array(self.x)

	# 定義域外の探索防止
	def check_domain(self):
		for i in range(self.cnf.prob_dim):
			self.x[i] = np.clip(self.x[i], self.fnc.axis_range[0], self.fnc.axis_range[1])
