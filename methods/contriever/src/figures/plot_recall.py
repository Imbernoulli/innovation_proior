import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

plt.rcParams.update({'xtick.direction': 'in'})
plt.rcParams.update({'ytick.direction': 'in'})
plt.rcParams.update({'legend.frameon': True})
#plt.rcParams.update({'grid.color': '#dddddd'})
plt.rcParams.update({'grid.linestyle': '--'})
plt.rcParams.update({'axes.axisbelow': True})
plt.rcParams["font.family"] = "Times New Roman"


FS = 13
MS=15
LW=4
LINESTYLE = ['', '', '-.', ':']
MARKERSTYLE=['o', 'X', '*', 'p', '.']
COLORS = ["xkcd:dark periwinkle", "xkcd:coral",'xkcd:goldenrod', "xkcd:jade green", "xkcd:coffee"]
#COLORS = ["xkcd:bright blue", "xkcd:coral",'xkcd:goldenrod']
legends = []

recall=0

##################
#plt.figure()
fig, axs = plt.subplots(1, sharex=True,figsize=(10, 2.5))

datasets= ['MSMARCO', 'Trec-COVID', 'NFCorpus', 'NQ', 'HotpotQA', 'FiQA', 'ArguAna', 'Tóuche-2020', 'Quora', 'CQADupStack', 'DBPedia', 'Scidocs', 'Fever', 'Climate-fever', 'Scifact']

bm25_recall = [65.8, 49.8, 25.0, 76.0, 74.0, 53.9, 94.2, 53.8, 97.3, 60.6, 39.8, 35.6, 93.1, 43.6, 90.8]
#ours_recall = [66.7, 12.8, 28.5, 76.2, 69.3, 56.8, 88.4, 27.4, 98.6, 43.6, 35.5, 93.6, 49.6, 92.4] ## ORIGINAL
ours_recall = [67.4, 21.1, 27.5, 77.6, 71.3, 53.7, 87.7, 23.9, 98.7, 60.5, 45.2, 35.4, 93.7, 48.9, 93.0] ## CORRECT MODEL
bert_recall = [3.5, 10.6, 6.7, 14.3, 15.8, 6.9, 59.1, 3, 74.6, 11.0, 7.1, 11.3, 13.6, 12.8, 35.2]
realm_recall = [52.6, 8.1, 23.0, 58.1, 56.1, 28, 73.1, 11.5, 92.7, 35.5, 33.0, 23.1, 82.6, 42.3, 83.8]
simcse_recall = [52.6, 23.9, 17.5, 40.0, 56.1, 40.3, 95.1, 17.5, 97.9, 47.3, 19.7, 22.0, 82.6, 42.3, 73.8]

idx = [i[0] for i in sorted(enumerate(ours_recall), key=lambda x:x[1])]

tmp = [ours_recall[i] for i in idx]
ours_recall= tmp
tmp = [bm25_recall[i] for i in idx]
bm25_recall= tmp
tmp = [bert_recall[i] for i in idx]
bert_recall= tmp
tmp = [realm_recall[i] for i in idx]
realm_recall = tmp
tmp = [simcse_recall[i] for i in idx]
simcse_recall = tmp
tmp = [datasets[i] for i in idx]
datasets= tmp

datasets.append('Avg.')
bm25_recall.append(mean(bm25_recall))
ours_recall.append(mean(ours_recall))
bert_recall.append(mean(bert_recall))
realm_recall.append(mean(realm_recall))
simcse_recall.append(mean(simcse_recall))

x = [i for i in range(len(ours_recall))]
x[len(x)-1] = x[len(x)-1]+1
w=.15

BERT = {
  "linestyle": LINESTYLE[2],
  "marker": MARKERSTYLE[2],
  "color": COLORS[2],
  'x': [i  for i in x],
  'recall': bert_recall,
  'labels': ['1M','32M','1B'],
  'label' : 'bert',
  'legend' : 'BERT',
}
#legends.append(BERT['legend'])
#axs.bar(BERT['x'], BERT['recall'],
#	width=w,
#	color=BERT['color'],
#	label=BERT['label'])

Realm = {
  "linestyle": LINESTYLE[0],
  "marker": MARKERSTYLE[0],
  "color": COLORS[3],
  'x': [i + w*0 for i in x],
  'recall':realm_recall,
  'label' : 'realm',
  'legend' : 'REALM',
}
legends.append(Realm['legend'])
axs.bar(Realm['x'], Realm['recall'],
	width=w,
	color=Realm['color'],
	label=Realm['label'])

SimCSE = {
  "linestyle": LINESTYLE[0],
  "marker": MARKERSTYLE[0],
  "color": COLORS[2],
  'x': [i + w*1 for i in x],
  'recall':simcse_recall,
  'label' : 'SimCSE',
  'legend' : 'SimCSE',
}
legends.append(SimCSE['legend'])
axs.bar(SimCSE['x'], SimCSE['recall'],
        width=w,
        color=SimCSE['color'],
        label=SimCSE['label'])

BM25 = {
  "linestyle": LINESTYLE[1],
  "marker": MARKERSTYLE[1],
  "color": COLORS[1],
  'x': [i+w*2  for i in x],
  'recall': bm25_recall,
  'label' : 'bm25',
  'legend' : 'BM25',
}
legends.append(BM25['legend'])
axs.bar(BM25['x'], BM25['recall'],
	width=w,
	color=BM25['color'],
	label=BM25['label'],
	 align='center')


Ours = {
  "linestyle": LINESTYLE[0],
  "marker": MARKERSTYLE[0],
  "color": COLORS[0],
  'x': [i + w*3 for i in x],
  'recall':ours_recall,
  'label' : 'ours',
  'legend' : 'Ours',
}
legends.append(Ours['legend'])
axs.bar(Ours['x'], Ours['recall'],
	width=w,
	color=Ours['color'],
	label=Ours['label'])

axs.set_ylabel('Recall@100', fontsize=FS)
axs.legend(legends, fontsize=FS, loc='upper left',bbox_to_anchor=[0,1], handlelength=1, ncol=5)

axs.grid()

xticks = [x[i] + 2*w for i in range(len(x))]
xticks[len(x)-1] -= w
xticklabels = datasets
axs.set_xticks(xticks)
plt.draw()
i=0
for tick in axs.get_xticklabels():
  i = i + 1
  tick.set_rotation(45)
  tick.set_ha('right')
  if i ==len(x)-1:
    break
axs.set_xticklabels(xticklabels, fontsize=FS)

yticks=[0, 50, 100]
axs.set_yticks(yticks)
axs.set_yticklabels([0,50,100],fontsize=FS)

a = axs.get_ygridlines()
#b = a[1]
#b.set_color('#332c13')

axs.set_ylim([0,100])

plt.savefig("figure_recall.pdf", bbox_inches='tight')
