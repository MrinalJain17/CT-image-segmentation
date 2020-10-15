class AttrDict(dict):
    """Subclasses dict and define getter-setter. This behaves as both dict and obj

    Taken from: https://github.com/facebookresearch/fair-sslime/blob/master/sslime/utils/collections.py
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
