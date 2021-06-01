import sys
import os
import csv
import collections
import random

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy
from sklearn import metrics, cluster, decomposition
from adjustText import adjust_text
from absl import app
from absl import flags

from wals_linear_regression import load_embeddings, load_wals

FLAGS = flags.FLAGS


def sidebyside_pca(names, embs, cats1, cats2):
	colors = ['#0F60FF','#02CEE8','#E81809','#14E802','#D1FF0F','#FFDC03','#E89909','#FF6103','#0FFF8E','#FF03D1']
	pca = decomposition.PCA(n_components = 2)
	red = pca.fit_transform(embs)
	fig,ax = plt.subplots(1, 2, figsize = (30,15))
	X = red[:,0]
	Y = red[:,1]
	ax[0].scatter(X, Y, c = [colors[l] for l in cats1])
	ax[1].scatter(X, Y, c = [colors[l] for l in cats2])

	for i, n in enumerate(names):
		ax[0].annotate(n, (X[i],Y[i]))
		ax[1].annotate(n, (X[i],Y[i]))
	fig.tight_layout()

	# patches = [mpatches.Patch(color = cmap[g], label = g) for g in cmap]
	# plt.legend(handles = patches)
	plt.show()


def colorshape_pca(names, embs, cats1, cats2, legend1, savefile):
	colors = ['#0F60FF','#02CEE8','#E81809','#14E802','#FF03D1','#FFDC03','#E89909','#FF6103','#0FFF8E','#D1FF0F']
	shapes = ['o', 's', '^', '*', 'P', 'D', '2', 'h', 'x', '+']
	pca = decomposition.PCA(n_components = 2)
	red = pca.fit_transform(embs)
	fig,ax = plt.subplots(figsize = (12,6))
	X = red[:,0]
	Y = red[:,1]
	for i in range(len(X)):
		ax.scatter(X[i], Y[i], s = 75, c = colors[cats1[i]], marker = shapes[cats2[i]])

	annotations = []
	for i, n in enumerate(names):
		annotations.append(ax.text(X[i], Y[i], n, fontsize = 15))
	adjust_text(annotations)
	# adjust_text(annotations)
	# adjust_text(annotations)

	fig.tight_layout()

	patches = [mpatches.Patch(color = colors[i], label = legend1[i]) for i in range(len(legend1))]
	plt.legend(handles = patches)
	# plt.show()
	plt.savefig(savefile, pad_inches = 0)


def k_means(embs, categories):
	ncats = len(set(categories.values()))
	X = []
	Y = []
	for w in embs:
		X.append(embs[w])
		Y.append(categories[w])
	kmeans = cluster.KMeans(n_clusters = 4, n_jobs = -1, n_init = 50, max_iter = 1000).fit(X)
	print(metrics.adjusted_rand_score(Y, kmeans.labels_))
	return kmeans.labels_

def spectral_cluster(embs, categories, n_clusters = 4):
	ncats = len(set(categories.values()))
	X = []
	Y = []
	for w in embs:
		X.append(embs[w])
		Y.append(categories[w])
	spectral = cluster.SpectralClustering(n_clusters = n_clusters, n_jobs = -1, n_init = 50, assign_labels='discretize').fit(X)
	# print(metrics.adjusted_rand_score(Y, spectral.labels_))
	return spectral.labels_, metrics.adjusted_rand_score(Y, spectral.labels_)

def centroid_distances(embs, categories):
	pass


def ttest_1samp(data, popmean, alternative = 'two-sided'):
	eps = 1e-8
	n = len(data)
	m = np.mean(data)
	s = np.std(data, ddof = 1)
	t = (m - popmean)/(s / np.sqrt(n) + eps)

	left = stats.t.cdf(t, n - 1)
	right = 1 - left
	if alternative == 'two-sided':
		p = min(left, right) * 2
	elif alternative == 'greater':
		p = right
	elif alternative == 'less':
		p = left
	else:
		p = None
	return t, p


def random_pred(langembs, walsmat, n_cats):
	random.seed()
	results = []
	for i in range(n_cats):
		lresult = []
		for hol in langembs:
			y = []
			for l in langembs:
				if l != hol and walsmat[l][i]!='_':
					y.append(walsmat[l][i])

			if walsmat[hol][i] in y:
				lbl = random.choice(y)
				if lbl == walsmat[hol][i]:
					lresult.append(1)
				else:
					lresult.append(0)
			else:
				lresult.append(-3)

		results.append(lresult)
	results = np.transpose(results)
	return results

def plot_dendrogram(langembs):
	langs = list(langembs.keys())
	embs = [langembs[l] for l in langs]
	langs = [wals2eng[l] for l in langs]
	Z = hierarchy.linkage(embs, 'ward')
	fig,ax = plt.subplots(figsize = (13,6))
	# fig.tight_layout()
	dng = hierarchy.dendrogram(Z, ax = ax, labels = langs)
	plt.show()

def _get_col_acc(col, frac = False):
	total = 0
	correct = 0
	for i in col:
		if i >= 0:
			total += 1
			correct += i
	if frac:
		return correct, total
	else:
		return correct/total


def area_accuracies(results, permtest = 0, frac = False, show_header = True):
	amap = dict()
	areas = set()
	with open(FLAGS.wals_areas) as fin:
		for line in fin:
			wc, a = line.strip().split('\t')
			amap[wc] = a
			areas.add(a)
	areas = sorted(list(areas))

	with open(results) as cin:
		creader = csv.reader(cin)
		header = []
		mat = []
		for row in creader:
			if len(header) == 0:
				header = row[1:]
			else:
				mat.append([float(x) for x in row[1:]])

	def _get_area_accs(mat, header, frac = False):
		area_acc = dict()
		for i in areas:
			area_acc[i] = [0,0]
		mat = np.asarray(mat)
		for i,wcat in enumerate(header):
			curr = wcat.split()[0]
			ccat = amap[curr]
			acc = _get_col_acc(mat[:,i], frac = frac)
			if frac:
				area_acc[ccat][0]+=acc[0]
				area_acc[ccat][1]+=acc[1]
			else:
				area_acc[ccat][0]+=acc
				area_acc[ccat][1]+=1
		return area_acc

	if permtest > 0:
		wiki2wals = {'en': 'eng', 'ar': 'ams', 'bg': 'bul', 'ca': 'ctl', 'hr': 'scr', 'cs': 'cze', 'da': 'dsh', 'nl': 'dut', 'et': 'est', 'fi': 'fin', 'fr': 'fre', 'de': 'ger', 'el': 'grk', 'he': 'heb', 'hu': 'hun', 'id': 'ind', 'it': 'ita', 'no': 'nor', 'pl': 'pol', 'pt': 'por', 'ro': 'rom', 'ru': 'rus', 'sk': 'svk', 'sl': 'slo', 'es': 'spa', 'sv': 'swe', 'tr': 'tur', 'uk': 'ukr', 'vi': 'vie'}
		langs = list(wiki2wals.values())
		walsheader, walsmat = load_wals(FLAGS.wals, langs, .5)

		precord = dict()
		for i in areas:
			precord[i] = []

		for i in range(permtest):
			sys.stdout.write(f'\r{i+1}/{permtest}')
			pmat = random_pred(langs, walsmat, len(walsheader))
			pacc = _get_area_accs(pmat, walsheader, frac = frac)
			for ar in areas:
				if pacc[ar][1]!=0:
					precord[ar].append(pacc[ar][0]/pacc[ar][1])
			if i == 0:
				nacc = dict()
				for ar in precord:
					if precord[ar] != []:
						nacc[ar] = precord[ar]
				precord = nacc

		print()
		ob_acc = _get_area_accs(mat, header, frac = frac)
		res = dict()
		for ar in precord:
			obs = ob_acc[ar][0] / ob_acc[ar][1]
			res[ar] = (ar, obs, stats.percentileofscore(precord[ar], obs, kind = 'strict'))
			print('{}:\tmean: {}\tpercentile:{}'.format(*res[ar]))
			print(np.mean(precord[ar]), np.std(precord[ar], ddof = 1))
	else:
		area_acc = _get_area_accs(mat, header, frac = frac)
		l1 = []
		l2 = []
		res = []
		for ar in areas:
			if area_acc[ar][1] != 0:
				l1.append(ar)
				if frac:
					l2.append(str(area_acc[ar][0]) + '/' + str(area_acc[ar][1]))
				else:
					# print(f'{ar}:\t{area_acc[ar][0]/area_acc[ar][1]}')
					l2.append(area_acc[ar][0]/area_acc[ar][1])
		if show_header:
			print('\t'.join(l1))
		if frac:
			print('\t'.join(l2))
		else:
			print('\t'.join(map(str, l2)))
			return l2


def area_acc_more_stats(results_list, baseline, savefile = 'area_acc_results.csv'):
	amap = dict()
	areas = set()
	with open(FLAGS.wals_areas) as fin:
		for line in fin:
			wc, a = line.strip().split('\t')
			amap[wc] = a
			areas.add(a)
	areas = sorted(list(areas))

	n_results = len(results_list)
	expres = [dict() for i in range(n_results+1)]
	for i, nm in enumerate(results_list + [baseline]):
		with open(nm) as cin:
			creader = csv.reader(cin)
			header = []
			mat = []
			for row in creader:
				if len(header) == 0:
					header = [s.split()[0] for s in row[1:]]
				else:
					mat.append([float(x) for x in row[1:]])
		mat = np.asarray(mat)
		for c, h in enumerate(header):
			expres[i][h] = _get_col_acc(mat[:,c], frac = False)

	baseline = expres[-1]
	expres = expres[:-1]

	alpha = .01
	table = {a:[0,[],0,0,0] for a in areas} # n, baseline, mean, p, n significant above baseline
	expcats = [{a:[] for a in areas} for i in range(n_results)]
	for cat in baseline:
		if cat not in amap:
			print(f'Category {cat} not found in area classification')
			continue
		ar = amap[cat]
		table[ar][0] += 1
		table[ar][1].append(baseline[cat])
		ex = []
		for i in range(n_results):
			ex.append(expres[i][cat])
			expcats[i][ar].append(expres[i][cat])
		t, p = ttest_1samp(ex, baseline[cat], alternative = 'greater')
		if p < alpha:
			table[ar][4]+=1
		if ar == 'Syntax':
			print(cat, baseline[cat], np.mean(ex))

	for ar in areas:
		table[ar][1] = np.mean(table[ar][1])
		ex = []
		for i in range(n_results):
			ex.append(np.mean(expcats[i][ar]))
		table[ar][2] = np.mean(ex)
		t, p = ttest_1samp(ex, table[ar][1], alternative = 'greater')
		table[ar][3] = p

	with open(savefile, 'w') as fout:
		headers = ['n', 'baseline', 'mean', 'p', 'n sig above']
		fout.write('\t'.join(['Areas'] + areas) + '\n')
		for i in range(len(headers)):
			outs = [headers[i]]
			for ar in areas:
				outs.append(str(table[ar][i]))
			fout.write('\t'.join(outs) + '\n')


def cluster_and_plot():
	global wals2eng
	wiki2wals = {'en': 'eng', 'ar': 'ams', 'bg': 'bul', 'ca': 'ctl', 'hr': 'scr', 'cs': 'cze', 'da': 'dsh', 'nl': 'dut', 'et': 'est', 'fi': 'fin', 'fr': 'fre', 'de': 'ger', 'el': 'grk', 'he': 'heb', 'hu': 'hun', 'id': 'ind', 'it': 'ita', 'no': 'nor', 'pl': 'pol', 'pt': 'por', 'ro': 'rom', 'ru': 'rus', 'sk': 'svk', 'sl': 'slo', 'es': 'spa', 'sv': 'swe', 'tr': 'tur', 'uk': 'ukr', 'vi': 'vie'}
	# wiki2wals = {'ar': 'ams', 'bg': 'bul', 'ca': 'ctl', 'hr': 'scr', 'cs': 'cze', 'da': 'dsh', 'nl': 'dut', 'et': 'est', 'fi': 'fin', 'fr': 'fre', 'de': 'ger', 'el': 'grk', 'hu': 'hun', 'id': 'ind', 'it': 'ita', 'pl': 'pol', 'pt': 'por', 'ro': 'rom', 'ru': 'rus', 'sk': 'svk', 'sl': 'slo', 'es': 'spa', 'sv': 'swe', 'tr': 'tur', 'uk': 'ukr', 'vi': 'vie'}
	wals2eng = {'eng' : 'English', 'ams' : 'Arabic', 'bul' : 'Bulgarian', 'ctl' : 'Catalan', 'scr' : 'Croatian', 'cze' : 'Czech', 'dsh' : 'Danish', 'dut' : 'Dutch', 'est' : 'Estonian', 'fin' : 'Finnish', 'fre' : 'French', 'ger' : 'German', 'grk' : 'Greek', 'heb' : 'Hebrew', 'hun' : 'Hungarian', 'ind' : 'Indonesian', 'ita' : 'Italian', 'nor' : 'Norwegian', 'pol' : 'Polish', 'por' : 'Portuguese', 'rom' : 'Romanian', 'rus' : 'Russian', 'svk' : 'Slovak', 'slo' : 'Slovene', 'spa' : 'Spanish', 'swe' : 'Swedish', 'tur' : 'Turkish', 'ukr' : 'Ukrainian', 'vie' : 'Vietnamese'}
	langembs = load_embeddings(FLAGS.embeddings)
	nl = dict()
	for l in wiki2wals:
		if l in langembs:
			nl[wiki2wals[l]] = langembs[l]
	langembs = nl

	# plot_dendrogram(langembs)

	names = list(langembs.keys())
	header, walsd = load_wals(FLAGS.wals, names, startfrom = 6)
	genus = dict()
	g2id = dict()
	for w in names:
		if walsd[w][0] not in g2id:
			g2id[walsd[w][0]] = len(g2id)
		genus[w] = g2id[walsd[w][0]]

	pred, rand = spectral_cluster(langembs, genus, 4)
	# sidebyside_pca(names, [langembs[w] for w in names], pred, [genus[w] for w in names])
	colorshape_pca([wals2eng[n] for n in names], [langembs[w] for w in names], [genus[w] for w in names], pred, [y[0] for y in sorted(g2id.items(), key = lambda x: x[1])], FLAGS.savefigure)


def validate_models():
	if FLAGS.baseline:
		area_accuracies(FLAGS.baseline, permtest = 0, frac = False)
	if FLAGS.baseline:
		area_accuracies(FLAGS.results, permtest = 0, frac = False)

def main():
	if FLAGS.func == 'cluster':
		cluster_and_plot()
	elif FLAGS.func == 'validate':
		validate_models()

if __name__ == '__main__':
	flags.DEFINE_string('embeddings', None, 'Language embeddings location', short_name = 'e')
	flags.DEFINE_string('wals', 'languages_all_fields.csv', 'WALS data')
	flags.DEFINE_string('baseline', None, 'Generated baseline')
	flags.DEFINE_string('results', None, 'Generated results to summarize')
	flags.DEFINE_string('savefigure', None, 'Where to save generated figure')
	flags.DEFINE_string('wals_areas', 'wals_custom_areas.tsv', 'File that defines the WALS areas')
	flags.DEFINE_enum('func', 'cluster', ['cluster', 'validate'], 'Which function to run')

	app.run(main)

	
	# modeldir = 'models'
	# area_acc_more_stats([f'{modeldir}/t_{i+1}/results.csv' for i in range(0,100)], f'{modeldir}/baseline.csv', f'area_acc_{modeldir}.csv')