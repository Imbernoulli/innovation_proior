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


FS = 15
MS=15
LW=4
LINESTYLE = ['', '', '-.', ':']
MARKERSTYLE=['o', 'X', '*', 'p', '.']
COLORS = ["xkcd:dark periwinkle", "xkcd:coral",'xkcd:goldenrod']
legends = []

ndcg=1
recall=0

##################
#plt.figure()
fig, axs = plt.subplots(2, sharex=True,figsize=(10, 5))

datasets= ['msmarco', 'Trec-COVID', 'NFCorpus', 'NQ', 'HotpotQA', 'FiQA', 'ArguAna', 'Tóuche-2020', 'Quora', 'DBPedia', 'Scidocs', 'Fever', 'Climate-fever', 'Scifact', 'Avg.']

bm25 = [22.8, 65.6, 32.5, 32.9, 60.3, 23.6, 31.5, 36.7, 78.9, 31.3, 15.8, 75.3, 21.3, 66.5]
bm25.append(mean(bm25))
#ours = [19.7, 23.2, 31.0, 26.3, 46.7, 25.1, 36.4, 17.4, 80.3, 25.5, 14.7, 62.2, 17.7, 58.7]
ours = [20.8, 32.4, 29.7, 25.9, 48.6, 23.3, 35.6, 19.7, 83.6, 29.6, 14.3, 68.1, 17.5, 63.9] ## CORRECT MODEL
ours.append(mean(ours))
bert=[0, 9.5, 2.7, 2.7, 5.2, 1.2, 23.7, 2.5, 47.8, 3.6, 2.3, 4.5, 2.7, 6.7]
bert.append(mean(bert))

bm25_recall = [65.8, 49.8, 25, 76, 74, 53.9, 94.2, 53.8, 97.3, 39.8, 35.6, 93.1, 43.6, 90.8]
bm25_recall.append(mean(bm25_recall))
#ours_recall = [66.7, 12.8, 28.5, 76.2, 69.3, 56.8, 88.4, 27.4, 98.6, 43.6, 35.5, 93.6, 49.6, 92.4]
ours_recall = [67.4, 21.1, 27.5, 77.6, 71.3, 53.7, 87.7, 23.9, 98.7, 45.2, 35.4, 93.7, 48.9, 93.0] ## CORRECT MODEL
ours_recall.append(mean(ours_recall))
bert_recall=[3.5, 10.6, 6.7, 14.3, 15.8, 6.9, 59.1, 3, 74.6, 7.1, 11.3, 13.6, 12.8, 35.2]
bert_recall.append(mean(bert_recall))


x = [i for i in range(len(ours))]
x[len(x)-1] = x[len(x)-1]+1
w=.2

BERT = {
  "linestyle": LINESTYLE[2],
  "marker": MARKERSTYLE[2],
  "color": COLORS[2],
  'x': [i  for i in x],
  'ndcg': bert,
  'recall': bert_recall,
  'labels': ['1M','32M','1B'],
  'label' : 'bert',
  'legend' : 'BERT',
}
legends.append(BERT['legend'])
axs[ndcg].bar(BERT['x'], BERT['ndcg'], 
	width=w,
	color=BERT['color'],
	label=BERT['label'])

axs[recall].bar(BERT['x'], BERT['recall'], 
	width=w,
	color=BERT['color'],
	label=BERT['label'])

BM25 = {
  "linestyle": LINESTYLE[1],
  "marker": MARKERSTYLE[1],
  "color": COLORS[1],
  'x': [i+w  for i in x],
  'ndcg': bm25,
  'recall': bm25_recall,
  'label' : 'bm25',
  'legend' : 'BM25',
}
legends.append(BM25['legend'])
axs[ndcg].bar(BM25['x'], BM25['ndcg'], 
	width=w,
	color=BM25['color'],
	label=BM25['label'],
	 align='center')
axs[recall].bar(BM25['x'], BM25['recall'], 
	width=w,
	color=BM25['color'],
	label=BM25['label'],
	 align='center')


Ours = {
  "linestyle": LINESTYLE[0],
  "marker": MARKERSTYLE[0],
  "color": COLORS[0],
  'x': [i + w*2 for i in x],
  'ndcg': ours,
  'recall':ours_recall,
  'label' : 'ours',
  'legend' : 'Ours',
}
legends.append(Ours['legend'])
axs[ndcg].bar(Ours['x'], Ours['ndcg'], 
	width=w,
	color=Ours['color'],
	label=Ours['label'])
axs[recall].bar(Ours['x'], Ours['recall'], 
	width=w,
	color=Ours['color'],
	label=Ours['label'])




axs[ndcg].set_ylabel('nDCG@10', fontsize=FS)
axs[recall].set_ylabel('Recall@100', fontsize=FS)
axs[1].legend(legends, fontsize=FS,loc='upper left',framealpha=1,bbox_to_anchor=[0.16,1.21], handlelength=1, ncol=4)

axs[ndcg].grid()
axs[recall].grid()

xticks = [x[i] + 2*w for i in range(len(x))]
xticks[len(x)-1] -= w
xticklabels = datasets
axs[1].set_xticks(xticks)
plt.draw()
i=0
for tick in axs[1].get_xticklabels():
  i = i + 1
  tick.set_rotation(45)
  tick.set_ha('right')
  if i ==len(x)-1: 
    break
axs[1].set_xticklabels(xticklabels, fontsize=FS)

yticks=[0, 50, 100]
axs[ndcg].set_yticks(yticks)
axs[recall].set_yticks(yticks)
axs[ndcg].set_yticklabels([0,50,100],fontsize=FS)
axs[recall].set_yticklabels([0,50,100],fontsize=FS)

a = axs[recall].get_ygridlines()
#b = a[1]
#b.set_color('#332c13')

axs[ndcg].set_ylim([0,100])
axs[recall].set_ylim([0,100])

plt.savefig("figure_zeroshot.pdf", bbox_inches='tight')
