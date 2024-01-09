import numpy as np

class RandomRuleGenerator:
    """
    An iterable to just generate a random int from 0 - 11 indicating
    which feature should be the next rule

    It follows statistics from the monkey version of the task,
    where there's 50 50 chance of an intra vs extra dimensional shift
    """
    def __init__(self, seed, num_cards=4, num_dims=3):
        self.rng = np.random.default_rng(seed)
        self.num_rules = num_cards*num_dims
        self.num_dims = num_dims
        self.num_rules_per_dim = self.num_rules // self.num_dims

        self.rule = self.rng.integers(0, self.num_rules, 1)[0]
        self.dimension = self.rule // self.num_rules_per_dim


    def __iter__(self):
        return self

    def __next__(self): 
        """
        randomly generates integer from 0 - 11
        """
        shift_type = self.rng.choice([0,1])
        # If 0, intra dimensional shift
        # If 1, extra dimensional shift

        if shift_type==0:
            self.dimension = self.dimension
        else:
            dimensions = np.arange(self.num_dims)
            dimensions = dimensions[dimensions!=self.dimension]
            self.dimension = self.rng.choice(dimensions)

        features = np.arange(self.num_rules)
        features = features[features//self.num_rules_per_dim==self.dimension]
        features = features[features!=self.rule]
        self.rule = self.rng.choice(features)
        
        return self.rule