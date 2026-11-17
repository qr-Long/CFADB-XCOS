"""Figure 6: Fe XXIII similarity matrices, without matching markers."""
from pathlib import Path
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from core.data import load_pair_from_h5
from core.similarity import compute_xcos_parts
from viewer.threshold_viewer import _make_tick_positions_labels

def draw(pair,sim,upper,output):
    rows=np.flatnonzero(pair['info1']['idx'].to_numpy()<=upper) if upper else np.arange(sim.shape[0])
    cols=np.flatnonzero(pair['info2']['idx'].to_numpy()<=upper) if upper else np.arange(sim.shape[1])
    view=sim[np.ix_(rows,cols)]
    cmap=ListedColormap(['#3A4750','#048998','#3bb4c1','#9CDEC9','#D6FFF2'])
    bounds=np.linspace(view.min(),view.max(),6)
    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(view,aspect='auto',interpolation='nearest',cmap=cmap,norm=BoundaryNorm(bounds,cmap.N,clip=True))
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,boundaries=bounds)
    for axis,info,indices in [(ax.xaxis,pair['info2'],cols),(ax.yaxis,pair['info1'],rows)]:
        pos,labels=_make_tick_positions_labels(info,indices,label_mode='idx',max_tick_labels=10,force_all=False)
        axis.set_ticks(pos); axis.set_ticklabels(labels,fontsize=8,rotation=90 if axis is ax.xaxis else 0)
    ax.set_xlabel(pair['db2']); ax.set_ylabel(pair['db1'])
    fig.tight_layout(); fig.savefig(output,bbox_inches='tight'); plt.close(fig)

def main():
    (HERE/'photos').mkdir(parents=True,exist_ok=True)
    pair=load_pair_from_h5('042623','atomdb313','spex30801',seq='be-like')
    sim,_=compute_xcos_parts(pair)
    for upper,suffix in [(None,'a_fe_xxiii_similarity_full'),(100,'b_fe_xxiii_similarity_zoom')]:
        output=HERE/'photos'/('fig06'+suffix+'.pdf'); draw(pair,sim,upper,output); print(output)

if __name__ == '__main__':
    main()
