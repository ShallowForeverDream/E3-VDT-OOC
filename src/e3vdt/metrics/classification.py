from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable, List

def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    yt=list(y_true); yp=list(y_pred)
    return sum(a==b for a,b in zip(yt,yp))/len(yt) if yt else 0.0
def precision_recall_f1(y_true: List[str], y_pred: List[str], label: str) -> Dict[str,float]:
    tp=sum(t==label and p==label for t,p in zip(y_true,y_pred)); fp=sum(t!=label and p==label for t,p in zip(y_true,y_pred)); fn=sum(t==label and p!=label for t,p in zip(y_true,y_pred))
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0; f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"precision":precision,"recall":recall,"f1":f1,"support":sum(t==label for t in y_true)}
def classification_report(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str,object]:
    yt=list(y_true); yp=list(y_pred); labels=sorted(set(yt)|set(yp)); per_label={label:precision_recall_f1(yt,yp,label) for label in labels}; macro_f1=sum(v["f1"] for v in per_label.values())/len(per_label) if per_label else 0.0
    return {"accuracy":accuracy(yt,yp),"macro_f1":macro_f1,"labels":per_label,"n":len(yt),"label_distribution_true":dict(Counter(yt)),"label_distribution_pred":dict(Counter(yp))}
