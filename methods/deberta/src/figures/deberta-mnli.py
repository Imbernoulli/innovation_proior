import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import *
import matplotlib
import numpy as np

style=['r-s','c-p', 'b:*', 'm:o', 'k:v']
name = ["150k", "250k", "350k", "450k", "550k", "650k", "750k", "850k", "1M"]

bert_rip = [82.4,82.9,83.7,84.1,84.3,84.5,84.6,84.8, 85]
deberta = [83.9,84.8,85.2,85.6,85.8,85.9,86.1,86.2, 86.3]
bert = [84.5] * len(deberta)
roberta = [84.7] * len(deberta)
xlnet = [85.0] * len(deberta)

index = list(range(0, len(bert_rip)))
plt.ylabel('Accuracy (MNLI)', fontsize=28)
plt.xlabel('Number of pre-training steps', fontsize=28)


plt.plot(index, bert_rip, style[0], markersize=14, linewidth=3)
plt.plot(index, deberta, style[1], markersize=14, linewidth=3)
#plt.plot(index, bert, style[2], markersize=14, linewidth=3)
#plt.plot(index, roberta, style[3], markersize=14, linewidth=3)
#plt.plot(index, xlnet, style[4], markersize=14, linewidth=3)

plt.axis([0, 8, 81, 86.5])
plt.legend([r'RoBERTa-ReImp$_{base}$', r'DeBERTa$_{base}$'], loc="lower right", fontsize=28)
#plt.legend([r'RoBERTa-ReImp$_{base}$', r'DeBERTa$_{base}$', r'BERT$_{base}$', r'RoBERTa$_{base}$'], loc="lower right", fontsize=28)
plt.ticklabel_format(style='sci' ,axis='y', labelsize=28)
plt.tick_params(labelsize=28)
plt.xticks(index, name)
plt.show()
