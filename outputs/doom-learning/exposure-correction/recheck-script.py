from pathlib import Path
import numpy as np,json
from doom_learning.common import OUT,save_json,capture_provenance
from doom_learning.brain import MemoryBrain
from doom_learning.controls import shifted_exposure
from doom_learning.run import episode
out=OUT/'exposure-correction';out.mkdir(exist_ok=False);capture_provenance(out)
(out/'recheck-script.py').write_text(Path(__file__).read_text())
b=MemoryBrain();rows=[]
for seed in [41031,41032]:
    paired=json.loads((OUT/'survival-pilot'/f'{seed}-plastic'/'train-1.json').read_text())
    horizon=round(paired['horizon_seconds']*35)
    original=np.zeros(horizon,dtype=bool);original[:len(paired['trace'])]=[r['US_active'] for r in paired['trace']]
    shifted,offset=shifted_exposure(original,paired['game_tics'],seed+700000)
    b.reset();r,actual=episode(b,seed,paired['horizon_seconds'],learning=True,stimulation_schedule=shifted)
    save_json(out/f'{seed}.json',r)
    rows.append({'seed':seed,'paired_exposure_tics':int(original.sum()),'corrected_delivered_tics':int(actual.sum()),'dose_matched':int(actual.sum())==int(original.sum()),'shift_tics':offset,'changed_edges':r['after']['changed_edges'],'survival_seconds':r['survival_seconds']})
    print(json.dumps(rows[-1]),flush=True)
save_json(out/'results.json',{'scope':'Recheck corrected US scheduling only; not a replacement survival-learning study. Original failed pilot preserved.','rows':rows})
