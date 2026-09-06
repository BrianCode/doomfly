"""Reproducible R8 pathway assay; no game score or learned decoder involved."""
import argparse,time
import numpy as np
from doom_learning.common import annotations,OUT,save_json,digest


def frame_for(label,width=320,height=240):
    f=np.zeros((height,width,3),dtype=np.uint8)
    if label=='black':return f
    if label=='green':f[:,:,1]=255
    elif label=='white':f[:]=255
    else:
        f[:,:,2]=255
        if label=='left_blue':f[:,width//2:]=0
        elif label=='right_blue':f[:,:width//2]=0
        elif label=='vertical':f[:,(np.arange(width)//40)%2==0]=0
        elif label=='horizontal':f[(np.arange(height)//30)%2==0,:]=0
        elif label!='blue':raise ValueError(label)
    return f


def run(revision='v2',out=None):
    if revision=='v2':from doom_learning_v2.visual import VisualMemoryBrain
    elif revision=='v3':from doom_learning_v3.visual import VisualMemoryBrain
    elif revision=='v4':from doom_learning_v4.visual import VisualMemoryBrain
    else:raise ValueError(revision)
    b=VisualMemoryBrain();a=annotations(b.ids)
    groups={t:np.flatnonzero(a.type.eq(t)) for t in ['aMe12','R8p','R8y','KCg-d','MBON11','PPL101','DNp20','DNpe017']}
    groups['all_KCs']=b.circuit['kc']
    out=out or OUT/f'physiology-{revision}/rgb-vision-reproducible.json';rows=[]
    for label in ['black','blue','green','white','left_blue','right_blue','vertical','horizontal']:
        b.reset();total=np.zeros(b.n,dtype=np.int64);wall=0;trace=[];frame=frame_for(label)
        for t in range(35):
            c,w=b.rgb_step(frame,(round((t+1)*10000/35)-b.cursor)*.1);total+=c;wall+=w
            trace.append({k:int(c[ix].sum()) for k,ix in groups.items()})
        row={'label':label,'wall':wall,'groups':{k:{'spikes':int(total[ix].sum()),'active':int(np.count_nonzero(total[ix]))} for k,ix in groups.items()},
             'kc_counts':total[groups['KCg-d']].tolist(),'trace':trace,'input_sha256':digest(frame)}
        rows.append(row);print({k:v for k,v in row.items() if k not in ['kc_counts','trace']},flush=True)
        save_json(out,{'revision':revision,'visual_model':b.visual_report,'conditions':rows,'complete':label=='horizontal','biologically_validated':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',choices=['v2','v3','v4'],default='v2');p.add_argument('--out');args=p.parse_args();run(args.revision,args.out)
