import matplotlib.pyplot as plt
import sys
from matplotlib.pyplot import *
import matplotlib
import numpy as np

style=['r-s','c-p', 'b:*', 'm:o', 'k:v']
name = ["150k", "250k", "350k", "450k", "550k", "650k", "750k", "850k", "1M"]

bert_rip = [75.2, 76.9, 77, 77.5, 78.3, 78.4, 78.7, 79.2, 79.5]
deberta = [78.9, 79.2, 79.6, 79.8, 80.9, 81.2, 81.7, 81.9, 82.2]
bert = [76.3] * len(deberta)
roberta = [79.7] * len(deberta)
xlnet = [81.3] * len(deberta)

index = list(range(0, len(bert_rip)))
plt.ylabel('F1 (SQuAD v2.0)', fontsize=28)
plt.xlabel('Number of pre-training steps', fontsize=28)


plt.plot(index, bert_rip, style[0], markersize=14, linewidth=3)
plt.plot(index, deberta, style[1], markersize=14, linewidth=3)
#plt.plot(index, bert, style[2], markersize=14, linewidth=3)
#plt.plot(index, roberta, style[3], markersize=14, linewidth=3)
#plt.plot(index, xlnet, style[4], markersize=14, linewidth=3)

plt.axis([0, 8, 73, 83])
plt.legend([r'RoBERTa-ReImp$_{base}$', r'DeBERTa$_{base}$'], loc="lower right", fontsize=28)
#plt.legend([r'RoBERTa-ReImp$_{base}$', r'DeBERTa$_{base}$', r'BERT$_{base}$', r'RoBERTa$_{base}$', r'XLNet$_{base}$'], loc="lower right", fontsize=28)
plt.ticklabel_format(style='sci' ,axis='y', labelsize=28)
plt.tick_params(labelsize=28)
plt.xticks(index, name)
plt.show()
