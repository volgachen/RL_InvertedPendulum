import json

default_args = {}

class Argument(object):
    def __init__(self, filename):
        self.args = default_args.copy()
        new_args = json.load(open(filename, 'r'))
        assert type(new_args) == dict
        self.args.update(new_args)
    
    def __getitem__(self, key):
        assert key in self.args.keys(), 'Key %s is not initialized.'%(key)
        return self.args[key]

    def __getattr__(self, key):
        assert key in self.args.keys(), 'Key %s is not initialized.'%(key)
        return self.args[key]

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.args, f)