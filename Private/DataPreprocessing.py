



def str_to_int(_strs):
    """
    good5 = ['N', 'V', '/', 'L', 'R']
    N: normal
    V: premature ventricular contraction
    /: paced beats
    L: left bundle branch block
    R: right bundle branch block
    """
    _len = range(len(_strs))
    for i in _len:
        if _strs[i]  == 'N':
            _strs[i] = 0
        elif _strs[i] == 'V':
            _strs[i] = 1
        elif _strs[i] == '/':
            _strs[i] = 2
        elif _strs[i] == 'L':
            _strs[i] = 3
        elif _strs[i] == 'R':
            _strs[i] = 4
    return list(map(int, _strs))


def save_data(_data):
    pass
    