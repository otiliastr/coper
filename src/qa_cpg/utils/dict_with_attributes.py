class AttributeDict(object):
    def __init__(self, d):
        # Convert all nested dictionaries into AttrDict.
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = AttributeDict(v)
        # Convert d to AttrDict.
        self.__dict__ = d
