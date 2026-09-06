import numpy as np
import pytest

from doom.connectome import exact_ids, index_edges


def test_large_neuron_ids_keep_every_bit():
    ids = ["720575940000000001", "720575940000000002"]
    assert exact_ids(ids).tolist() == [720575940000000001, 720575940000000002]
    with pytest.raises(ValueError, match="never floats"):
        exact_ids([float(ids[0])])


def test_no_weak_edge_or_autapse_pruning():
    ids = exact_ids([10, 20, 30])
    i, j, count, keep = index_edges(ids, [10, 20, 30], [20, 20, 10], [1, 2, 500])
    assert keep.tolist() == [True, True, True]
    assert i.tolist() == [0, 1, 2]
    assert j.tolist() == [1, 1, 0]
    assert count.tolist() == [1, 2, 500]


def test_unknown_endpoints_accounted_for_without_aliasing():
    ids = exact_ids([10, 30])
    i, j, count, keep = index_edges(ids, [1, 10, 20, 30, 99], [10, 30, 30, 99, 10], [1, 2, 3, 4, 5])
    assert keep.tolist() == [False, True, False, False, False]
    assert i.tolist() == [0] and j.tolist() == [1] and count.tolist() == [2]


@pytest.mark.parametrize("count", [[0], [-1], [1.5], [np.nan], [2**32]])
def test_invalid_synapse_count_rejected(count):
    with pytest.raises(ValueError, match="Synapse counts"):
        index_edges(exact_ids([1, 2]), [1], [2], count)


def test_duplicate_node_ids_fail_before_import():
    with pytest.raises(ValueError, match="unique"):
        index_edges(exact_ids([1, 1]), [1], [1], [1])

def test_explicit_male_glia_excluded_even_if_superclass_is_assigned():
    import pandas as pd
    from doom.connectome import normalize_nodes
    frame=pd.DataFrame({'bodyId':[1,2,3],'superclass':['cb_intrinsic','cb_intrinsic',None],
      'statusLabel':['Reviewed']*3,'type':['test']*3,'status':['Glia','Traced','Orphan']})
    catalog,nodes=normalize_nodes('malecns_v1',frame)
    assert nodes.source_id.tolist()==[2]
    assert catalog.loc[catalog.source_id.eq(1),'object_kind'].item()=='non_neuronal'


from doom.transmitters import transmitter_signs

def test_ambiguous_transmitters_are_preserved_and_sensitivity_is_local():
    names=['acetylcholine','gaba','histamine','glutamate','acetylcholine,dopamine',None,'dopamine','acetylcholine,gaba']
    positive, uncertain=transmitter_signs(names,1)
    negative,_=transmitter_signs(names,-1)
    assert positive.tolist()==[1,-1,-1,-1,1,1,1,1]
    assert uncertain.tolist()==[False]*5+[True]*3
    np.testing.assert_array_equal(positive[~uncertain],negative[~uncertain])
    assert np.all(positive[uncertain]==-negative[uncertain])
    with pytest.raises(ValueError): transmitter_signs(names,0)
