from cap_anndata import CapAnnDataUns


def test_basic_init():
    d = {'key1': 'value1', 'key2': {'value2': 'sub_value2'}}
    cap_d = CapAnnDataUns(d)
    assert cap_d == d
    assert cap_d.keys_to_remove == []
    assert cap_d.get("key1") == "value1"
    assert cap_d.get("key_N") == None


def test_pop():
    cap_d = CapAnnDataUns()
    cap_d["key1"] = "value1"
    cap_d["key2"] = "value2"

    cap_d.pop("key1")
    
    assert len(cap_d.keys()) == 1
    assert cap_d.keys_to_remove == ["key1"]

    cap_d["key1"] = "new_value"

    assert len(cap_d.keys()) == 2
    assert cap_d.keys_to_remove == []

    cap_d.popitem()
    assert len(cap_d.keys()) == 1
    assert cap_d.keys_to_remove == ["key1"]

    cap_d["key1"] = {'sk1': 'v1', "sk2": 'v2'}
    cap_d["key1"].pop('sk2')

    assert len(cap_d.keys()) == 2
    assert cap_d.keys_to_remove == []
