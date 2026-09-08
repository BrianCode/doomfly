"""Re-label a server checkpoint for another kernel backend or source revision.

The neural arrays are shared between the CPU and cuTile backends (the GPU
brain exports the CPU delay-queue format), so a checkpoint only fails to resume
because its recorded identity (kernel build record and source hashes) no longer
matches the running process. This tool rewrites exactly that identity, verifies
the checksums and the model configuration signature against a freshly
constructed brain, and writes a new checkpoint generation. The run record is
kept and marked as a conversion so the audit trail shows the backend change.
"""
import argparse,json,re,shutil,uuid
from pathlib import Path
import numpy as np
from doom.checkpoint import digest
from doom.provenance import provenance
ROOT=Path(__file__).resolve().parents[1]
GRAPH=ROOT/'outputs/doom/malecns_v1/graph.npz'


def load_generation(directory):
    pointer=json.loads((directory/'latest.json').read_text())['generation']
    if not re.fullmatch('[a-f0-9]{32}',pointer):raise ValueError('Invalid checkpoint identifier')
    path=directory/pointer
    state=json.loads((path/'state.json').read_text())
    for name,expected in state['sha256'].items():
        if digest(path/name)!=expected:raise ValueError(f'Checksum mismatch for {name}')
    return path,state


def kernel_build(backend):
    from doom.native import BUILD as native
    if backend=='native':return native
    from doom.cutile_brain import BUILD as cutile
    return {**native,'backend':'cutile','cutile':cutile,'model_revision':cutile['model_revision']}


def convert(directory,model,backend,eta=.001):
    directory=Path(directory)
    source,state=load_generation(directory)
    identity=json.loads(json.dumps(state['identity']))
    origin=provenance(GRAPH,kernel_build(backend),identity['provenance']['assets'])
    brain=None
    if model=='experimental-v6':
        from doom_learning_v6.calibration import calibrated_brain
        from doom.training import candidate_provenance
        brain=calibrated_brain(eta=eta,backend=backend)
        origin['candidate']=candidate_provenance(brain,ROOT)
        origin['model_revision']='adaptive-centered-v6-live-v1'
        origin['kernel']=brain.build
    identity['provenance']=origin
    generation=uuid.uuid4().hex
    stage=directory/(generation+'.partial');stage.mkdir()
    try:
        if brain is not None:
            with np.load(source/'brain.npz',allow_pickle=False) as a:
                metadata=json.loads(str(a['metadata']))
                if metadata['configuration_sha256']!=brain.configuration_signature():
                    raise ValueError('Checkpoint configuration differs from the target model; refusing to relabel')
                arrays={k:a[k] for k in a.files if k!='metadata'}
            metadata['build']=brain.build
            metadata['converted_from_build']=json.loads(str(np.load(source/'brain.npz',allow_pickle=False)['metadata']))['build']
            np.savez(stage/'brain.npz',metadata=json.dumps(metadata),**arrays)
        else:
            shutil.copy2(source/'brain.npz',stage/'brain.npz')
        for name in state['sha256']:
            if name!='brain.npz':shutil.copy2(source/name,stage/name)
        record=dict(state['record']);record['recovery']=f'converted-to-{backend}-backend';record['converted_from_generation']=source.name
        data={**state,'identity':identity,'record':record,'sha256':{name:digest(stage/name) for name in state['sha256']}}
        (stage/'state.json').write_text(json.dumps(data,indent=2)+'\n')
        stage.rename(directory/generation)
        pointer=directory/'latest.partial'
        pointer.write_text(json.dumps({'generation':generation})+'\n');pointer.replace(directory/'latest.json')
        return generation
    except Exception:
        shutil.rmtree(stage,ignore_errors=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--model',choices=['baseline','experimental-v6'],default='experimental-v6')
    p.add_argument('--backend',choices=['native','cutile'],default='cutile')
    p.add_argument('--eta',type=float,default=.001)
    a=p.parse_args()
    print(json.dumps({'generation':convert(a.checkpoint_dir,a.model,a.backend,a.eta),'backend':a.backend,'model':a.model}))


if __name__=='__main__':main()
