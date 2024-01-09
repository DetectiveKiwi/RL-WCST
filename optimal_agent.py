import numpy as np

class OptimalAgent:
    """
    Implements an optimal agent for this version of the WCST.
    This agent maximizes the probability of choosing the correct card based on history.
    
    Essentially, it keeps track of all possible rules (1 for each). If a feature is tested and is incorrect, value is set to 0
    The agent chooses the card with the most amount of 1's
    
    Note, it does not take advantage of rule switches
    If all features are set to 0, then the rule vector is reset
    """

    def __init__(self, seed=None, num_cards=4, num_dims=3):
        self.rng = np.random.default_rng(seed)
        self.num_cards = num_cards
        self.num_dims = num_dims

        self.reset()

    def reset(self):
        self.possible_rules = np.ones(self.num_cards * self.num_dims, dtype=int)
        self.chosen_card = None

    def make_selection(self, cards):
        """
        Given a set of cards, makes a selection based on attention and 
        Args:
            cards: np array of 4 x 3, 4 cards by 3 indexes of features
        Returns:
            int from 0 - 3, representing card index of choice
        """
        card_values = np.empty(self.num_cards)
        
        for i,card in enumerate(cards):
            card_values[i] = np.sum(self.possible_rules[card])
        
        card_idx = np.random.choice(np.argwhere(card_values==np.max(card_values))[:,0])
        
        self.chosen_card = cards[card_idx]

        return card_idx

    def evaluate_feedback(self, is_correct):
        """
        Given feedback about previous choice, update attentions
        Args:
            feedback, tuple of (is_correct: bool, value: float)
        """
        if is_correct:
            mask = np.zeros(self.num_cards * self.num_dims, dtype=int)
            mask[self.chosen_card] = 1
            self.possible_rules = self.possible_rules * mask
        else:
            self.possible_rules[self.chosen_card] = 0
            
            if np.sum(self.possible_rules)==0:
                self.possible_rules = np.ones(self.num_cards * self.num_dims, dtype=int)
                self.possible_rules[self.chosen_card] = 0