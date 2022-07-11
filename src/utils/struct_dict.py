from collections import UserDict


class structdict(UserDict):
    def __init__(self, a_dict: dict):
        super().__init__(a_dict)
        [setattr(self, k, v) for k, v in a_dict.items()]

    @classmethod
    def from_keys_values(cls, keys=[], values=[]):
        return cls(dict(zip(keys, values)))

    def __setattr__(self, name, val):
        if name == "data":
            super().__setattr__(name, val)
        else:
            self[name] = val
            self.__dict__[name] = val

    def __setitem__(self, key, item):
        super(structdict, self).__setitem__(key, item)
        self.__dict__[key] = item
