"""Exact cell identities and existing connections for a candidate memory rule."""
import numpy as np
from .common import annotations, digest


def identify(brain):
    a=annotations(brain.ids); types=a.type.fillna('')
    kc=np.flatnonzero(types.str.startswith('KC')).astype(np.int32)
    mb=np.flatnonzero(types.eq('MBON11')).astype(np.int32)
    dan=np.flatnonzero(types.eq('PPL101')).astype(np.int32)
    if len(mb)!=2 or len(dan)!=2:raise ValueError('Expected two MBON11 and two PPL101 cells in this MaleCNS release.')
    edges=np.flatnonzero(np.isin(brain.post,mb)).astype(np.int64)
    pre=np.searchsorted(brain.ptr,edges,side='right')-1
    keep=np.isin(pre,kc);edges=edges[keep];pre=pre[keep].astype(np.int32)
    if not len(edges) or np.any(brain.weight[edges]<=0):raise ValueError('Candidate KC-to-MBON inputs must be existing positive-weight edges.')
    # Anatomical pooling proxy. Direct PPL101->MBON11 contact weights specify
    # fractional participation, not measured dopamine release or receptor gain.
    contact=np.zeros((len(dan),len(mb)),dtype=np.float32)
    for d,i in enumerate(dan):
        for m,j in enumerate(mb):
            ix=np.flatnonzero(brain.post[brain.ptr[i]:brain.ptr[i+1]]==j)+brain.ptr[i]
            contact[d,m]=np.abs(brain.weight[ix]).sum()
    if np.any(contact.sum(axis=0)<=0):raise ValueError('Missing reconstructed DAN-to-MBON contact support.')
    gain=(contact/contact.sum(axis=0))[...,np.searchsorted(mb,brain.post[edges])].copy()
    mask=np.zeros(brain.n,dtype=np.uint8);mask[kc]=1
    dan_index=np.full(brain.n,-1,dtype=np.int8);dan_index[dan]=np.arange(len(dan))
    def cells(indices):return [{'index':int(i),'id':str(brain.ids[i]),'type':str(a.type.iloc[i]),
                               'instance':str(a.instance.iloc[i]),'soma_side':str(a.somaSide.iloc[i])} for i in indices]
    report={'dataset':'MaleCNS v1.0','neurons':brain.n,'edges':len(brain.weight),'kc_count':len(kc),
        'plastic_edges':len(edges),'plastic_edge_index_sha256':digest(edges),
        'plastic_presynaptic_index_sha256':digest(pre),'DAN':cells(dan),'MBON':cells(mb),
        'gamma_d_KC':cells(np.flatnonzero(types.eq('KCg-d'))),
        'dan_to_mbon_contact_fraction':(contact/contact.sum(axis=0)).tolist(),
        'selection':'All existing KC-to-MBON11 edges; no graph pruning or added connections.',
        'compartment_assignment':'Cell-type-level gamma1/peduncle inference; aggregate edges do not resolve receptor localization.',
        'modulation':'PPL101 outgoing edges deliver a transmitter trace to every recorded target. Only the specified KC-to-MBON11 efficacy response is modeled. Other target responses remain unknown.',
        'baseline_difference':'Two PPL101 cells use a modulatory channel rather than the baseline default fast excitatory sign.',
        'evidence':['https://doi.org/10.1016/j.neuron.2015.11.003','https://doi.org/10.7554/eLife.16135','https://male-cns.janelia.org/download/']}
    return {'kc':kc,'mb':mb,'dan':dan,'edges':edges,'pre':pre,'gain':gain.astype(np.float32),
            'kc_mask':mask,'dan_index':dan_index,'report':report}
