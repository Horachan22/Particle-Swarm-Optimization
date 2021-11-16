'''
===================================================================
Project Name     : Differential Evolution
File Name        : main.py
Encoding         : UTF-8
Creation Date    : 2021/10/19
===================================================================
'''

import numpy           as np
import configuration   as cf
import function        as fc
import optimizer       as op
import logger          as lg

def run(opt, cnf, fnc, log):
	opt.initialize_solutions()                  # 初期化
	log.logging(opt.pop, fnc.total_evals)       # 初期個体群ログ
	while fnc.total_evals < cnf.max_evals:      # 評価回数上限まで実行
		opt.get_next_population()               # 次世代個体群生成
		log.logging(opt.pop, fnc.total_evals)   # 次世代個体群ログ
	log.outLog(opt.pop, fnc.total_evals)        # ログ出力(trial'n'.csv)

def main():
	cnf = cf.Configuration()                                        # configurationインスタンス生成
	for i in range(len(cnf.prob_name)):                             # 関数の個数だけ探索
		log = lg.Logger(cnf, cnf.prob_name[i])                      # loggerインスタンス生成
		fnc = fc.Function(cnf.prob_name[i], cnf.prob_dim)           # 探索する関数のインスタンス生成
		for j in range(cnf.max_trial):                              # 試行回数まで実行
			fnc.resetTotalEvals()                                   # 総評価回数(functionクラス内変数)リセット
			cnf.setRandomSeed(seed = j + 1)                         # ランダムシード値設定
			opt = op.DifferentialEvolution(cnf, fnc)                # optimizerインスタンス生成
			run(opt, cnf, fnc, log)                                 # 探索実行
		sts = lg.Statistics(cnf, fnc, log.path_out, log.path_trial) # 関数ごとの統計を作成
		sts.outStatistics()                                         # 統計出力(all_trials.csv, statistics.csv)

if __name__ == '__main__':
	main()