import numpy as np
import pandas as pd

from card_generators import RandomCardGenerator
from rule_generators import RandomRuleGenerator


class WcstSession:
    """
    A configurable session for the WCST. Features are given by numbers (e.g. star=0, triangle=1, etc.)
    
    Allowable configurations include
     - correct and incorrect value amounts
     - number of cards,
     - number of dimensions,
     - number of correct trials to criterion or running average of trials to criterion (None for infinite)
    """
    def __init__(
        self,
        num_cards=4,
        num_dims=3,
        trials_to_crit1=[8,8],
        trials_to_crit2=[16,20],
        random_seed=None
    ):
        """
        Args:
            num_cards: number of cards to use
            num_dims: number of dimensions to use
            trials_to_crit1 (two element list): number of correct trials (elem 0) out of previous number of trials (elem 1)
                Use 'None' to indicate no rule switch
            trials_to_crit2 (two element list): number of correct trials (elem 0) out of previous number of trials (elem 1)
                Use 'None' to indicate no rule switch
            random seed: seed used to initialize random generators 
        """
        self.num_cards = num_cards
        self.num_dims = num_dims
        self.trials_to_crit1 = trials_to_crit1
        self.trials_to_crit2 = trials_to_crit2
        
        self.card_generator = RandomCardGenerator(random_seed, self.num_cards, self.num_dims)
        self.rule_generator = RandomRuleGenerator(random_seed, self.num_cards, self.num_dims)
        
        self.start_new_session()
    
    def start_new_session(self):
        """
        Starts a new session of WCST, wipes any history or tracking
        """
        self.card_iterator = iter(self.card_generator)
        self.rule_iterator = iter(self.rule_generator)
        
        self.history = []
        self.block_perf = []
        self.current_rule = next(self.rule_iterator)
        self.current_trial = 0
        self.trial_in_block = 0
        self.current_block = 0
        self.trial_reward = 0
        self.total_rewards = 0
        self.current_cards = None
        self.current_selection = None
        self.generated_cards_for_trial = False
        self.is_correct = None

    def get_cards(self):
        """
        Get the cards to display for trial
        Returns: np array of num_cards x num_dimensions
        """
        if not self.generated_cards_for_trial:
            self.current_cards = next(self.card_generator)
            self.generated_cards_for_trial = True
        return self.current_cards

    def make_selection(self, selection):
        """
        Makes a selection of a card, logs information about the trial,
        checks whether to update the rule/block, moves on to next trial. 

        Args:
            selection: int of 0, 1, 2, 3. Index of card 
        Returns:
            outcome: where outcome is bool for Correct/Incorrect
        """
        if self.current_cards is None:
            raise ValueError("No current cards on screen, call get_cards() first")
        self.current_selection = selection
        card = self.current_cards[selection]
        is_correct = self.current_rule in card
        self.block_perf.append(is_correct)
        self.is_correct = is_correct

        self._log_trial()

        if self.block_switch_condition():
            self.block_perf = []
            self.current_block += 1
            self.current_rule = next(self.rule_iterator)
            self.trial_in_block = 0
        else:
            self.trial_in_block += 1
        
        self.current_cards = None
        self.generated_cards_for_trial = False
        self.current_trial += 1
        return is_correct

    def block_switch_condition(self):
        """
        Block switching condition for monkeys
        Returns true if at least one trials_to_crit is satisfied
            False otherwise
        """
        block_perf_history = np.array(self.block_perf)
        
        if self.trials_to_crit1 is not None:
            if np.count_nonzero(
                block_perf_history[-self.trials_to_crit1[1]:]
            ) >= self.trials_to_crit1[0]:
                return True
        if self.trials_to_crit2 is not None:
            if np.count_nonzero(
                block_perf_history[-self.trials_to_crit2[1]:]
            ) >= self.trials_to_crit2[0]:
                return True
            
        return False

    def _log_trial(self):
        """
        Helper func to log trial information in history
        """
        row = {
            "TrialNumber": self.current_trial,
            "BlockNumber": self.current_block,
            "TrialAfterRuleChange": self.trial_in_block,
            "Response": "Correct" if self.is_correct else "Incorrect",
            "ItemChosen": self.current_selection,
            "CurrentRule": self.current_rule
        }
        for card_idx, card in enumerate(self.current_cards):
            for dim_idx in range(self.num_dims):
                row[f"Item{card_idx} Dim{dim_idx}"] = card[dim_idx]
        self.history.append(row)


    def dump_history(self):
        """
        Creates a dataframe of current history in session, 
        """
        return pd.DataFrame(self.history)
