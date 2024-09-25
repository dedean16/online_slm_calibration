import h5py

def get_dict_from_hdf5(group: h5py.Group) -> dict:
    """
    Retrieve a hdf5 file or group as a dictionary. Supports nested dictionaries. Can be used on the main group as well
    to get a dictionary of the whole hdf5 structure.

    Args:
        group: hdf5 group.

    Returns: the group as dictionary
    """
    dic = {}
    for key in group.keys():
        if group[key].__class__.__name__ == 'Group':
            dic[key] = get_dict_from_hdf5(group[key])
        else:
            dic[key] = group[key][()]
    return dic
