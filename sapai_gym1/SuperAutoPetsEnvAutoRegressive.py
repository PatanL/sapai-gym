import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List
import itertools
from sklearn.preprocessing import OneHotEncoder
import copy  # For deep copying opponent and agent states

# Import game-specific classes and data from game engine
from sapai import Player, Pet, Food, Battle, Shop, Team
from sapai.data import data 

# Import MaskablePPO for opponent models
from sb3_contrib import MaskablePPO 
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Define the number of action types (buy, sell, roll, etc.)
self.ACTION_TYPES = {
    "END_TURN": 0,
    "BUY_PET": 1,
    "BUY_FOOD": 2,
    "BUY_COMBINE": 3,
    "COMBINE": 4,
    "SELL": 5,
    "ROLL": 6,
    "REORDER": 7,
    # Potentially add FREEZE here if you implement it
}

# The number of discrete action types
num_action_types = len(self.ACTION_TYPES)

# The action space is now a dictionary
self.action_space = spaces.Dict({
    # The agent first chooses an action type
    "action_type": spaces.Discrete(num_action_types),

    # Then it chooses the first parameter (e.g., a shop slot or first team pet)
    # The size should be the max possible number of slots. 7 for shop, 5 for team. Let's use 7.
    "param1": spaces.Discrete(7),

    # And the second parameter (e.g., a team slot)
    "param2": spaces.Discrete(5)
})

class SuperAutoPetsEnv(gym.Env):
    """
    A Gymnasium environment for training a single-agent AI in Super Auto Pets using self-play with an opponent pool.
    This environment ensures mutual visibility between the agent and the opponent:
        - The agent sees its current state and the opponent's previous state.
        - The opponent sees its current state and the agent's previous state.
    """
    metadata = {'render_modes': ['human']}
    
    # Define constants
    MAX_ACTIONS = 213
    ACTION_BASE_NUM = {
        "end_turn": 0,
        "buy_pet": 1,
        "buy_food": 7,
        "buy_combine": 17,
        "combine": 47,
        "sell": 57,
        "roll": 62,
        "buy_food_team": 63,
        "reorder": 65,
    }
    MAX_TURN = 25
    BAD_ACTION_PENALTY = -0.1
    MAX_TEAM_PETS = 5
    MAX_SHOP_PETS = 6
    MAX_SHOP_FOODS = 3
    ALL_PETS = [
        "pet-ant", "pet-beaver", "pet-beetle", "pet-bluebird", "pet-cricket", "pet-duck",
        "pet-fish", "pet-horse", "pet-ladybug", "pet-mosquito", "pet-otter", "pet-pig",
        "pet-sloth", "pet-bat", "pet-crab", "pet-dodo", "pet-dog", "pet-dromedary",
        "pet-elephant", "pet-flamingo", "pet-hedgehog", "pet-peacock", "pet-rat",
        "pet-shrimp", "pet-spider", "pet-swan", "pet-tabby-cat", "pet-badger",
        "pet-blowfish", "pet-caterpillar", "pet-camel", "pet-hatching-chick",
        "pet-giraffe", "pet-kangaroo", "pet-owl", "pet-ox", "pet-puppy", "pet-rabbit",
        "pet-sheep", "pet-snail", "pet-tropical-fish", "pet-turtle", "pet-whale",
        "pet-bison", "pet-buffalo", "pet-deer", "pet-dolphin", "pet-hippo",
        "pet-llama", "pet-lobster", "pet-monkey", "pet-penguin", "pet-poodle",
        "pet-rooster", "pet-skunk", "pet-squirrel", "pet-worm", "pet-chicken",
        "pet-cow", "pet-crocodile", "pet-eagle", "pet-goat", "pet-microbe",
        "pet-parrot", "pet-rhino", "pet-scorpion", "pet-seal", "pet-shark",
        "pet-turkey", "pet-cat", "pet-boar", "pet-dragon", "pet-fly", "pet-gorilla",
        "pet-leopard", "pet-mammoth", "pet-octopus", "pet-sauropod", "pet-snake",
        "pet-tiger", "pet-tyrannosaurus", "pet-zombie-cricket", "pet-bus",
        "pet-zombie-fly", "pet-dirty-rat", "pet-chick", "pet-ram",
        "pet-butterfly", "pet-bee"
    ]
    ALL_FOODS = [
        "food-apple", "food-honey", "food-cupcake", "food-meat-bone",
        "food-sleeping-pill", "food-garlic", "food-salad-bowl",
        "food-canned-food", "food-pear", "food-chili", "food-chocolate",
        "food-sushi", "food-melon", "food-mushroom", "food-pizza",
        "food-steak", "food-milk"
    ]
    ALL_STATUSES = [
        "status-weak", "status-coconut-shield", "status-honey-bee",
        "status-bone-attack", "status-garlic-armor", "status-splash-attack",
        "status-melon-armor", "status-extra-life", "status-steak-attack",
        "status-poison-attack"
    ]

    def __init__(self, opponent_generator=None, valid_actions_only=True, manual_battles=False):
        """
        Initialize the SuperAutoPetsEnv environment.
        
        :param opponent_generator: Not used in self-play. Set to None.
        :param valid_actions_only: If True, invalid actions will raise exceptions.
        :param manual_battles: If True, battles are controlled manually.
        """
        super(SuperAutoPetsEnv, self).__init__()

        # Validate parameters
        if manual_battles:
            assert opponent_generator is None, "Opponent generator must be None for manual battles."
        # Define action space for single agent
        self.action_space = spaces.Discrete(self.MAX_ACTIONS)

        # Calculate observation space length
        # (len(ALL_PETS) + 2 + len(ALL_STATUSES)) * MAX_TEAM_PETS for each agent
        # + (len(ALL_FOODS) + 1) * MAX_SHOP_FOODS for shop foods
        # + 5 for other stats
        # + MAX_ACTIONS for action masks (to be removed)
        len_obs_space = (len(self.ALL_PETS) + 2 + len(self.ALL_STATUSES)) * self.MAX_TEAM_PETS + \
                        (len(self.ALL_FOODS) + 1) * self.MAX_SHOP_FOODS + 5
        total_obs_size = len_obs_space * 2  # For both agent and opponent

        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(total_obs_size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1660,), dtype=np.float32)
        self.reward_range = (-1.0, 1.0)

        self.valid_actions_only = valid_actions_only
        self.manual_battles = manual_battles

        # Initialize the single agent and the opponent
        self.agent = Player()
        self.opponent = Player()

        # Initialize shops and teams
        self.shop_agent = Shop(turn=1, pack="StandardPack")
        self.shop_opponent = Shop(turn=1, pack="StandardPack")

        self.agent.shop = self.shop_agent
        self.opponent.shop = self.shop_opponent

        self.team_agent = Team(pack="StandardPack")
        self.team_opponent = Team(pack="StandardPack")

        self.agent.team = self.team_agent
        self.opponent.team = self.team_opponent

        # Initialize game state
        self.bad_action_reward_sum_agent = 0
        self.bad_action_reward_sum_opponent = 0
        self.done = False
        self.info = {}
        self.last_actions = [None]  # Only for the agent

        # Initialize opponent pool
        self.opponent_pool: List[MaskablePPO] = []  
        self.max_opponents = 10  # Maximum number of opponents in the pool

        # Initialize variables for consistent opponent selection and previous states
        self.current_opponent_model: Optional[PPO] = None  # Opponent selected for the current game
        self.previous_opponent_state: Optional[Team] = None  # Opponent's state from the previous turn
        self.previous_agent_state: Optional[Team] = None  # Agent's state from the previous turn

        # Initialize random state
        self.rs_random = np.random.RandomState(None)
        self.reset()

        self.reorder_count_agent = 0
        self.allow_additional_reorder_agent = False

        self.reorder_count_opponent = 0
        self.allow_additional_reorder_opponent = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to start a new game.
        
        :return: Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rs_random = np.random.RandomState(seed)
        else:
            self.rs_random = np.random.RandomState()

        # Reset players
        seed_state = self.rs_random.get_state()
        self.agent = Player(seed_state=seed_state)
        self.opponent = Player(seed_state=seed_state)

        # Reset shops
        self.shop_agent = Shop(turn=1, pack="StandardPack")
        self.shop_opponent = Shop(turn=1, pack="StandardPack")
        self.agent.shop = self.shop_agent
        self.opponent.shop = self.shop_opponent

        # Reset teams
        self.team_agent = Team(pack="StandardPack")
        self.team_opponent = Team(pack="StandardPack")
        self.agent.team = self.team_agent
        self.opponent.team = self.team_opponent

        # Perform initial shop roll
        self.shop_agent.roll()
        self.shop_opponent.roll()

        # Reset game state
        self.done = False
        self.info = {}
        self.bad_action_reward_sum_agent = 0
        self.bad_action_reward_sum_opponent = 0
        self.last_actions = [None]

        # Reset turn counter
        self.turn = 1  # Assuming turn starts at 1

        # Reset reorder counters and flags
        self.reorder_count_agent = 0
        self.allow_additional_reorder_agent = False

        self.reorder_count_opponent = 0
        self.allow_additional_reorder_opponent = False

        # Select a consistent opponent for this game from the opponent pool
        if self.opponent_pool:
            self.current_opponent_model = self.rs_random.choice(self.opponent_pool)
            print(f"Selected an opponent from the pool for this game.")
        else:
            self.current_opponent_model = None  # Use default opponent actions
            print("Opponent pool is empty. Using default opponent strategy.")

        # Initialize previous states as None
        self.previous_opponent_state = None
        self.previous_agent_state = None

        # Store the initial agent state as previous_agent_state for the opponent
        self.previous_agent_state = copy.deepcopy(self.agent.team)

        # Generate initial observation
        obs = self._encode_state()
        
        # Generate action mask for the agent
        action_mask = self._generate_action_mask(agent_idx=0)[None, :] 
        self.info['action_mask'] = action_mask

        return obs, self.info

    def step(self, action):
        """
        Execute one step in the environment with the agent's action and the opponent's action.
        
        :param action: Action for the single agent.
        :return: Observation, aggregated reward, done flag, truncated flag, and info dictionary.
        """
        if self.done:
            obs, info = self.reset()
            return obs, 0.0, self.done, False, info

        # Store Agent's Previous State before taking the action
        self.previous_agent_state = copy.deepcopy(self.agent.team)

        # Resolve the agent's action
        agent_action_name = self.resolve_action(agent_idx=0, action=action)

        # Resolve the opponent's action
        opponent_action = self.get_opponent_action()
        opponent_action_name = self.resolve_action(agent_idx=1, action=opponent_action)

        # Resolve action
        #     player_to_act = self.player
        #     action_to_play = self._avail_actions()[action]
        #     action_name = self._get_action_name(action_to_play).split(".")[-1]
        #     action_method = getattr(player_to_act, action_name)
        #     action_method(*action_to_play[1:])

        #     # If turn is ended, play an opponent
        #     if action_name == "end_turn" and not self.manual_battles:
        #         opponent = self.opponents[self.player.turn - 1]
        #         battle_result = Battle(self.player.team, opponent).battle()
        #         self._player_fight_outcome(battle_result)
        #         self.player.start_turn()
        # self.last_action = action

        # Execute battle phase if both have ended their turns
        try:
            if self.agent.turn_ended and self.opponent.turn_ended and not self.manual_battles:
                # Store the current opponent state before the battle
                self.previous_opponent_state = copy.deepcopy(self.opponent.team)

                # Conduct the battle
                battle_result = Battle(self.agent.team, self.opponent.team).battle()
                self._player_fight_outcome(battle_result)
                self.agent.start_turn()
                self.opponent.start_turn()
                self.turn += 1

                self.reorder_count_agent = 0
                self.allow_additional_reorder_agent = False

                self.reorder_count_opponent = 0
                self.allow_additional_reorder_opponent = False

                # Check for game termination
                self.done = self.is_done()
        except Exception as e:
            # If the sapai engine crashes for any reason, catch it.
            print(f"WARNING: sapai engine crashed with error: {e}. Terminating episode.")
            # End the episode immediately. The agent will get a neutral or slightly
            # negative reward for this truncated episode.
            self.done = True 

        # Assign rewards
        reward_agent = self.get_reward_agent()
        aggregated_reward = reward_agent  # Use only agent's reward

        # Generate observations
        obs = self._encode_state()

        # Generate action mask for the agent
        action_mask = self._generate_action_mask(agent_idx=0)[None, :]
        self.info['action_mask'] = action_mask

        # Reset bad action penalties
        self.bad_action_reward_sum_agent = 0
        self.bad_action_reward_sum_opponent = 0

        return obs, aggregated_reward, self.done, False, self.info

    def resolve_action(self, agent_idx: int, action: int):
        """
        Resolve the action for a specific agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :param action: Action to be executed.
        """
        if not isinstance(action, int):
            action = int(action)

        player = self.agent if agent_idx == 0 else self.opponent
        print("available actions: ", self._avail_actions(agent_idx).keys())
        action_to_play = self._avail_actions(agent_idx).get(action, None)
        if action_to_play is None:
            if self.valid_actions_only:
                raise RuntimeError(f"Agent {agent_idx} attempted invalid action {action}.")
            # Apply penalty for invalid action
            if agent_idx == 0:
                self.bad_action_reward_sum_agent += self.BAD_ACTION_PENALTY
            else:
                self.bad_action_reward_sum_opponent += self.BAD_ACTION_PENALTY
            return

        # Execute the action
        action_name = self._get_action_name(action_to_play).split(".")[-1]
        action_method = getattr(player, action_name)
        action_method(*action_to_play[1:])

        # logic for steak and melon reward
        if agent_idx == 0 and action_name == "buy_food":
            shop_idx = action_to_play[1]
            food_name = player.shop[shop_idx].obj.name
            if food_name in ["food-melon", "food-steak"]:
                self.step_reward += 0.1 * self.shaping_scale

        # Track last actions for the agent
        if agent_idx == 0:
            self.last_actions[0] = action
        
        if action_name == "reorder":
            if agent_idx == 0:
                self.reorder_count_agent += 1
            else:
                self.reorder_count_opponent += 1
        
        # if agent_idx == 0:
        #     if action_name == "buy_pet":
        #         # Assuming Player class has an attribute 'last_bought_pet'
        #         bought_pet = player.last_bought_pet  # You need to ensure this attribute exists
        #         if bought_pet in ["pet-monkey", "pet-giraffe"]:
        #             self.allow_additional_reorder_agent = True
        #     elif action_name in ["trigger_pile_faint_effect"]:  # Replace with actual action names
        #         # Replace with the correct action name(s) that require additional reorder
        #         self.allow_additional_reorder_agent = True

        return action_name

    def add_opponent(self, opponent_model: MaskablePPO):
        """
        Add a new opponent to the opponent pool.
        
        :param opponent_model: A trained PPO model to be used as an opponent.
        """
        if len(self.opponent_pool) >= self.max_opponents:
            removed_opponent = self.opponent_pool.pop(0)  # Remove the oldest opponent
            print(f"Removed oldest opponent from the pool.")
        self.opponent_pool.append(opponent_model)
        print(f"Added new opponent to the pool. Pool size is now {len(self.opponent_pool)}.")

    def get_opponent_action(self):
        """
        Get the action from the selected opponent.
        If no opponent is selected (opponent_pool is empty), use a default opponent action strategy.
        
        :return: Action chosen by the opponent.
        """
        if self.current_opponent_model is None:
            return self.default_opponent_action()

        # Opponent's observation includes its current state and the agent's previous state
        opponent_obs = self._encode_state(agent_idx=1)  # Encode opponent's observation including agent's previous state
        action_mask = self.info.get('action_mask', None)
        opponent_action, _states = self.current_opponent_model.predict(opponent_obs, action_masks=action_mask, deterministic=True)
        return opponent_action

    def default_opponent_action(self):
        """
        Define a default opponent action strategy (e.g., random actions).
        
        :return: Randomly selected valid action.
        """
        available_actions = list(self._avail_actions(agent_idx=1).keys())
        if not available_actions:
            return self.ACTION_BASE_NUM["end_turn"]
        return np.random.choice(available_actions)

    def is_done(self):
        """
        Determine if the game has ended.
        
        :return: Boolean indicating if the game is done.
        """
        return (self.agent.lives <= 0 or self.opponent.lives <= 0 or self.turn >= self.MAX_TURN)

    def get_reward_agent(self):
        """
        Calculate reward for the agent.
        
        :return: Agent's reward.
        """
        reward_agent = self.agent.wins / 7 + self.bad_action_reward_sum_agent
    
        if self.opponent.lives <= 0:
            reward_agent += 1
        return reward_agent

    def _avail_end_turn(self, agent_idx: int):
        """
        Get available 'end_turn' actions for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Dictionary of available 'end_turn' actions.
        """
        action_dict = {}
        action_num = self.ACTION_BASE_NUM["end_turn"]
        action_dict[action_num] = (self.agent.end_turn,) if agent_idx == 0 else (self.opponent.end_turn,)
        return action_dict

    def _avail_buy_pets(self, agent_idx: int):
        """
        Get available 'buy_pet' actions for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Dictionary of available 'buy_pet' actions.
        """
        action_dict = {}
        player = self.agent if agent_idx == 0 else self.opponent
        shop = self.shop_agent if agent_idx == 0 else self.shop_opponent
        team = self.team_agent if agent_idx == 0 else self.team_opponent

        if len(player.team) >= self.MAX_TEAM_PETS:
            return action_dict  # Cannot buy if team is full

        pet_index = 0
        for shop_idx, shop_slot in enumerate(player.shop):
            if shop_slot.slot_type == "pet":
                if shop_slot.cost <= player.gold:
                    action_num = self.ACTION_BASE_NUM["buy_pet"] + pet_index
                    action_dict[action_num] = (player.buy_pet, shop_idx)
                pet_index += 1
        return action_dict

    def _avail_buy_foods(self, agent_idx: int):
        action_dict = dict()
        action_dict = {}
        player = self.agent if agent_idx == 0 else self.opponent
        shop = self.shop_agent if agent_idx == 0 else self.shop_opponent
        team = self.team_agent if agent_idx == 0 else self.team_opponent

        if len(player.team) == 0:
            return action_dict

        food_index = 0
        for shop_idx, shop_slot in enumerate(player.shop):
            if shop_slot.slot_type == "food":
                if shop_slot.cost <= player.gold:
                    # Multi-foods (eg. salad, sushi, etc.)
                    food_effect = data["foods"][shop_slot.obj.name]["ability"]["effect"]
                    if shop_slot.obj.name == "food-canned-food" or ("target" in food_effect and "kind" in food_effect["target"] and food_effect["target"]["kind"] == "RandomFriend"):
                        action_num = self.ACTION_BASE_NUM["buy_food_team"] + food_index
                        action_dict[action_num] = (player.buy_food, shop_idx)
                    else:
                        # Single target foods (eg. apple, melon)
                        for team_idx, team_slot in enumerate(player.team):
                            if team_slot.empty:
                                continue
                            action_num = self.ACTION_BASE_NUM["buy_food"] + (food_index * self.MAX_TEAM_PETS) + team_idx
                            action_dict[action_num] = (player.buy_food, shop_idx, team_idx)
                food_index += 1
        return action_dict
    
    def _avail_buy_combine(self, agent_idx: int):
        action_dict = {}
        player = self.agent if agent_idx == 0 else self.opponent
        shop = self.shop_agent if agent_idx == 0 else self.shop_opponent
        team = self.team_agent if agent_idx == 0 else self.team_opponent    

        team_names = {}
        if len(player.team) == 0:
            return action_dict

        # Find pet names on team
        for team_idx, slot in enumerate(player.team):
            if slot.empty:
                continue
            if slot.pet.name not in team_names:
                team_names[slot.pet.name] = []
            team_names[slot.pet.name].append(team_idx)

        # Search through pets in the shop
        shop_pet_index = 0
        for shop_idx, shop_slot in enumerate(player.shop):
            if shop_slot.slot_type == "pet":
                # Can't combine if pet not already on team
                if shop_slot.obj.name not in team_names:
                    continue

                if shop_slot.cost <= player.gold:
                    for team_idx in team_names[shop_slot.obj.name]:
                        action_num = self.ACTION_BASE_NUM["buy_combine"] + (shop_pet_index * self.MAX_TEAM_PETS) + team_idx
                        action_dict[action_num] = (player.buy_combine, shop_idx, team_idx)
                shop_pet_index += 1

        return action_dict

    def _avail_team_combine(self, agent_idx: int):
        """
        Get available 'combine' actions for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Dictionary of available 'combine' actions.
        """
        action_dict = {}
        player = self.agent if agent_idx == 0 else self.opponent
        team = self.team_agent if agent_idx == 0 else self.team_opponent

        if len(player.team) <= 1:
            return action_dict

        team_names = {}
        for slot_idx, slot in enumerate(player.team):
            if slot.empty:
                continue
            if slot.pet.name not in team_names:
                team_names[slot.pet.name] = []
            team_names[slot.pet.name].append(slot_idx)

        for pet_name, indices in team_names.items():
            if len(indices) == 1:
                continue

            for idx0, idx1 in itertools.combinations(indices, r=2):
                indexes = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
                action_num = self.ACTION_BASE_NUM["combine"] + indexes.index((idx0, idx1))
                action_dict[action_num] = (player.combine, idx0, idx1)

        return action_dict

    
    def _avail_sell(self, agent_idx):
        """
        Get available 'sell' actions for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Dictionary of available 'sell' actions.
        """
        action_dict = {}
        player = self.agent if agent_idx == 0 else self.opponent
        team = self.team_agent if agent_idx == 0 else self.team_opponent
        
        # Only allow selling if there are 5 or more pets on the team
        # maybe only allow to sell if you have more than 2 gold?
        if len([slot for slot in player.team if not slot.empty]) >= 5:
            for team_idx, slot in enumerate(player.team):
                if slot.empty:
                    continue
                action_num = self.ACTION_BASE_NUM["sell"] + team_idx
                action_dict[action_num] = (player.sell, team_idx)
        
        return action_dict

    def _avail_roll(self, agent_idx):
        """
        Get available 'roll' actions for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Dictionary of available 'roll' actions.
        """
        action_dict = dict()
        player = self.agent if agent_idx == 0 else self.opponent
        shop = self.shop_agent if agent_idx == 0 else self.shop_opponent

        if player.gold > 1:
            action_dict[self.ACTION_BASE_NUM["roll"]] = (player.roll,)
        return action_dict

    def _avail_reorder(self, agent_idx):
        """
        Get available 'reorder' actions for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Dictionary of available 'reorder' actions.
        """
        action_dict = {}

        if agent_idx == 0:
            reorder_count = self.reorder_count_agent
            allow_additional_reorder = self.allow_additional_reorder_agent
            player = self.agent
            team = self.team_agent
        else:
            reorder_count = self.reorder_count_opponent
            allow_additional_reorder = self.allow_additional_reorder_opponent
            player = self.opponent
            team = self.team_opponent
        
        max_reorders = 1
        if allow_additional_reorder and reorder_count < 2:
            max_reorders = 2
        
        if reorder_count >= max_reorders:
            return action_dict  # No reorder actions available

        if player.gold > 0:
            return {}

        team_size = len(player.team)
        offset = self.ACTION_BASE_NUM["reorder"] + sum([math.factorial(k) - 1 for k in range(team_size)])
        perms = itertools.permutations(range(team_size))

        # Skip the do-nothing permutation
        next(perms)

        return {
            offset + k: (player.reorder, perm)
            for k, perm in enumerate(perms)
        }

    def _get_action_name(self, input_action):
        """
        Retrieve the action name from the action tuple.
        
        :param input_action: Tuple containing the action method and its parameters.
        :return: Name of the action method.
        """
        return str(input_action[0].__name__)

    def _avail_actions(self, agent_idx: int):
        """
        Get all available actions for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Dictionary mapping action numbers to action tuples.
        """
        end_turn_actions = self._avail_end_turn(agent_idx)
        buy_pet_actions = self._avail_buy_pets(agent_idx)
        buy_food_actions = self._avail_buy_foods(agent_idx)
        buy_combine_actions = self._avail_buy_combine(agent_idx)
        team_combine_actions = self._avail_team_combine(agent_idx)
        sell_actions = self._avail_sell(agent_idx)
        roll_actions = self._avail_roll(agent_idx)
        reorder_actions = self._avail_reorder(agent_idx)

        # Combine all actions
        all_avail_actions = {**end_turn_actions, **buy_pet_actions, **buy_food_actions,
                             **buy_combine_actions, **team_combine_actions,
                             **sell_actions, **roll_actions, **reorder_actions}

        # Validate action uniqueness
        total_action_len = len(end_turn_actions) + len(buy_pet_actions) + len(buy_food_actions) + \
                           len(buy_combine_actions) + len(team_combine_actions) + \
                           len(sell_actions) + len(roll_actions) + len(reorder_actions)
        assert total_action_len == len(all_avail_actions), "Duplicate or missing actions detected."

        return all_avail_actions

    def _is_valid_action(self, action: int) -> bool:
        """
        Check if the action is valid.
        
        :param action: Action number to validate.
        :return: True if valid, False otherwise.
        """
        return 0 <= action < self.MAX_ACTIONS

    def action_masks(self) -> List[np.ndarray]:
        """
        Generate action masks for both the agent and the opponent.
        
        :return: List containing action masks for the agent and the opponent.
        """
        masks_agent = np.zeros(self.MAX_ACTIONS, dtype=bool)
        masks_opponent = np.zeros(self.MAX_ACTIONS, dtype=bool)

        avail_actions_agent = self._avail_actions(agent_idx=0).keys()
        avail_actions_opponent = self._avail_actions(agent_idx=1).keys()

        masks_agent[list(avail_actions_agent)] = True
        masks_opponent[list(avail_actions_opponent)] = True

        # return [masks_agent, masks_opponent]
        return masks_agent

    def _encode_pets(self, pets: List[Pet]) -> List[np.ndarray]:
        """
        One-hot encode the list of pets.
        
        :param pets: List of Pet objects.
        :return: List of encoded pet arrays.
        """
        arrays_to_concat = []
        for pet in pets:
            if pet.name == "pet-none":
                arrays_to_concat.append(np.zeros(len(self.ALL_PETS)))
                arrays_to_concat.append(np.zeros(2))  # Placeholder for attack and health, change to -> arrays_to_concat.append(np.zeros((2,)))
                arrays_to_concat.append(np.zeros(len(self.ALL_STATUSES)))
            else:
                arrays_to_concat.append(self._encode_single(pet.name, self.ALL_PETS))
                arrays_to_concat.append(np.array([pet.attack / 50, pet.health / 50]))
                if pet.status == "none":
                    arrays_to_concat.append(np.zeros(len(self.ALL_STATUSES))) # -> arrays_to_concat.append(np.zeros((len(self.ALL_STATUSES),)))
                else:
                    arrays_to_concat.append(self._encode_single(pet.status, self.ALL_STATUSES))
        return arrays_to_concat

    def _encode_foods(self, foods: List[tuple]) -> List[np.ndarray]:
        """
        One-hot encode the list of foods.
        
        :param foods: List of tuples containing Food objects and their costs.
        :return: List of encoded food arrays.
        """
        arrays_to_concat = []
        for food_tuple in foods:
            (food, cost) = food_tuple
            if food.name == "food-none":
                arrays_to_concat.append(np.zeros(len(self.ALL_FOODS)))
                arrays_to_concat.append(np.zeros(1))  # Placeholder for cost, change to arrays_to_concat.append(np.zeros((1,)))
            else:
                arrays_to_concat.append(self._encode_single(food.name, self.ALL_FOODS))
                arrays_to_concat.append(np.array([cost / 3]))  # Normalize cost (assuming max cost is 3)
        return arrays_to_concat

    # def _get_shop_foods(self, agent_idx: int) -> List[tuple]:
    #     """
    #     Retrieve the list of food items in the shop for the specified agent.
        
    #     :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
    #     :return: List of tuples containing Food objects and their costs.
    #     """
    #     shop = self.shop_agent if agent_idx == 0 else self.shop_opponent
    #     food_slots = []
    #     for slot in shop.foods:
    #         if slot.name != "food-none":
    #             food_slots.append((Food(slot.name), slot.cost))
    #     return food_slots

    def _get_shop_foods(self, agent_idx: int):
        """
        Retrieve the list of food items in the shop for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: List of tuples containing Food objects and their costs.
        """
        shop = self.shop_agent if agent_idx == 0 else self.shop_opponent
        player = self.agent if agent_idx == 0 else self.opponent
        food_slots = []
        for slot in player.shop._slots: #change back to shop_slots
            if slot.slot_type == "food":
                food_slots.append((slot.obj, slot.cost)) #change to slot.item?
        return food_slots

    def _encode_state(self, agent_idx: Optional[int] = None) -> np.ndarray:
        """
        Encode the current state of the game.
        
        :param agent_idx: Index of the agent to encode state for (0 for agent, 1 for opponent). If None, encode for both.
        :return: Encoded state as a NumPy array.
        """
        if agent_idx is None:
            # Encode Agent's Current State
            # encoded_team_pets_agent = self._encode_pets([p for p in self.team_agent.pets])
            # Encode team
            encoded_team_pets_agent = self._encode_pets([p.pet for p in self.agent.team])
            # Encode shop
            shop_pets = self.agent.shop.pets
            shop_foods = self._get_shop_foods(agent_idx=0)

            # Pad to the maximum number of pets and foods that can be in a shop
            while len(shop_pets) < 6:
                shop_pets.append(Pet("pet-none"))
            while len(shop_foods) < 2:
                shop_foods.append((Food("food-none"), 0))

            encoded_shop_pets_agent = self._encode_pets(shop_pets)
            encoded_shop_foods_agent = self._encode_foods(shop_foods)
            stats_agent = self._encode_stats(self.agent)

            # Encode Opponent's Previous State
            if self.previous_opponent_state is not None:
                # encoded_team_pets_opponent = self._encode_pets([p for p in self.previous_opponent_state.pets])
                encoded_team_pets_opponent = self._encode_pets([p.pet for p in self.previous_opponent_state])

                # opponent_shop_pets = self.opponent.shop.pets
                # opponent_shop_foods = self._get_shop_foods(agent_idx=1)

                # while len(opponent_shop_pets) < 6:
                #     opponent_shop_pets.append(Pet("pet-none"))
                # while len(opponent_shop_foods) < 2:
                #     opponent_shop_foods.append((Food("food-none"), 0))

                # encoded_shop_pets_opponent = self._encode_pets(opponent_shop_pets)
                # encoded_shop_foods_opponent = self._encode_foods(opponent_shop_foods)
                stats_opponent = self._encode_stats_opponent(self.opponent)
            else:
                # Initialize with default values if no previous state exists
                encoded_team_pets_opponent = self._encode_pets([Pet("pet-none")] * self.MAX_TEAM_PETS)
                # encoded_shop_pets_opponent = self._encode_pets([Pet("pet-none")] * self.MAX_SHOP_PETS)
                # encoded_shop_foods_opponent = self._encode_foods([(Food("food-none"), 0)] * self.MAX_SHOP_FOODS)
                stats_opponent = self._encode_stats_opponent(Player())

            # Concatenate Agent and Opponent States
            state_agent = np.concatenate(encoded_team_pets_agent + encoded_shop_pets_agent + encoded_shop_foods_agent + [stats_agent])
            state_opponent = np.concatenate(encoded_team_pets_opponent + [stats_opponent])

            # Concatenate both agent and opponent states
            combined_state = np.concatenate([state_agent, state_opponent])

            # After encoding
            # print("Combined state length:", len(combined_state))
            # print("Combined state shape:", combined_state.shape)

            # Convert to float32
            final_state = combined_state.astype(np.float32)


            return final_state

        else:
            # Encode State for a Specific Agent (used for Opponent's Observation)
            if agent_idx == 1:
                # Opponent's current state

                encoded_team_pets_opponent = self._encode_pets([p.pet for p in self.opponent.team])

                opponent_shop_pets = self.opponent.shop.pets
                opponent_shop_foods = self._get_shop_foods(agent_idx=1)

                while len(opponent_shop_pets) < 6:
                    opponent_shop_pets.append(Pet("pet-none"))
                while len(opponent_shop_foods) < 2:
                    opponent_shop_foods.append((Food("food-none"), 0))

                encoded_shop_pets_opponent = self._encode_pets(opponent_shop_pets)
                encoded_shop_foods_opponent = self._encode_foods(opponent_shop_foods)
                stats_opponent = self._encode_stats(self.opponent)

                # Agent's previous state
                if self.previous_agent_state is not None:
                    encoded_team_pets_agent_prev = self._encode_pets([p.pet for p in self.previous_agent_state])
                    stats_agent_prev = self._encode_stats_opponent(self.agent)
                else:
                    # Initialize with default values if no previous state exists
                    encoded_team_pets_agent_prev = self._encode_pets([Pet("pet-none")] * self.MAX_TEAM_PETS)
                    stats_agent_prev = self._encode_stats_opponent(Player())

                # Concatenate Opponent's and Agent's Previous States
                state_opponent = np.concatenate(encoded_team_pets_opponent + encoded_shop_pets_opponent + encoded_shop_foods_opponent + [stats_opponent])
                state_agent_prev = np.concatenate(encoded_team_pets_agent_prev + [stats_agent_prev])

                # Concatenate both opponent's current state and agent's previous state
                combined_specific_state = np.concatenate([state_opponent, state_agent_prev])

                # Convert to float32
                final_specific_state = combined_specific_state.astype(np.float32)

                return final_specific_state
            else:
                raise ValueError("Invalid agent_idx. Only agent_idx=0 (agent) and agent_idx=1 (opponent) are supported.")

    def _encode_single(self, value: str, category: List[str]) -> np.ndarray:
        """
        One-hot encode a single categorical value.
        
        :param value: The categorical value to encode.
        :param category: List of all possible categories.
        :return: One-hot encoded array.
        """
        encoder = OneHotEncoder(categories=[category], sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(np.array([[value]]))
        return encoded.flatten()

    def render(self, mode='human'):
        """
        Render the current state of the game.
        """
        print("=== Agent ===")
        self._render_player(self.agent)
        print("Shop Agent:")
        self._render_shop(self.shop_agent)
        print("=== Opponent ===")
        self._render_player(self.opponent)
        print("Shop Opponent:")
        self._render_shop(self.shop_opponent)
        print(f"Turn: {self.turn}")
        print(f"Game Over: {self.done}")

    def _render_player(self, player: Player):
        """
        Render the player's team and stats.
        
        :param player: Player object.
        """
        print(f"Wins: {player.wins}, Lives: {player.lives}, Gold: {player.gold}, Turn: {player.turn}")
        print("Team:")
        for pet in player.team.pets:
            print(f" - {pet.name} (Atk: {pet.attack}, HP: {pet.health}, Status: {pet.status})")

    def _render_shop(self, shop: Shop):
        """
        Render the shop's pets and foods.
        
        :param shop: Shop object.
        """
        print("Pets in Shop:")
        for pet in shop.pets:
            print(f" - {pet.name}")
        print("Foods in Shop:")
        for food, cost in shop.foods:
            print(f" - {food.name} (Cost: {cost})")

    def _player_fight_outcome(self, outcome: int):
        """
        Assign fight outcomes based on the battle result.
        
        :param outcome: Result of the battle (0: Agent wins, 1: Opponent wins, 2: Draw).
        """
        if outcome == 0:
            self.agent.lf_winner = True
            self.opponent.lf_winner = False
            self.agent.wins += 1
            self.opponent.lives -= 1
            self.opponent.lives = max(self.opponent.lives, 0)
        elif outcome == 1:
            self.opponent.lf_winner = True
            self.agent.lf_winner = False
            self.opponent.wins += 1
            self.agent.lives -= 1
            self.agent.lives = max(self.agent.lives, 0)
        elif outcome == 2:
            # Handle draw if necessary
            self.agent.lf_winner = True
            self.opponent.lf_winner = True

    def close(self):
        """
        Clean up resources.
        """
        pass

    def _generate_action_mask(self, agent_idx: int) -> np.ndarray:
        """
        Generate a binary action mask for the specified agent.
        
        :param agent_idx: Index of the agent (0 for agent, 1 for opponent).
        :return: Binary action mask as a NumPy array.
        """
        mask = np.zeros(self.MAX_ACTIONS, dtype=np.float32)
        available_actions = self._avail_actions(agent_idx).keys()
        mask[list(available_actions)] = True
        return mask

    def _encode_stats(self, player: Player) -> np.ndarray:
        """
        Encode player's stats.
        
        :param player: Player object.
        :return: Normalized stats array.
        """
        wins_normalized = player.wins / 10  # Assuming max wins is 10
        lives_normalized = player.lives / 6  # Assuming max lives is 10
        gold_normalized = min(player.gold, 30) / 30  # Assuming max gold is 30
        turn_normalized = min(player.turn, self.MAX_TURN) / self.MAX_TURN  # Assuming max turn is MAX_TURN
        shop_attack_normalized = min(player.shop.shop_attack, 20) / 20  # Assuming max shop attack is 20

        return np.array([
            wins_normalized,
            lives_normalized,
            gold_normalized,
            turn_normalized,
            shop_attack_normalized
        ])
    
    def _encode_stats_opponent(self, opponent: Player) -> np.ndarray:
        """
        Encode player's stats.
        
        :param player: Player object.
        :return: Normalized stats array.
        """
        wins_normalized = opponent.wins / 10  # Assuming max wins is 10
        lives_normalized = opponent.lives / 6  # Assuming max lives is 6
        turn_normalized = min(opponent.turn, self.MAX_TURN) / self.MAX_TURN  # Assuming max turn is MAX_TURN

        return np.array([
            wins_normalized,
            lives_normalized,
            turn_normalized
        ])
    
    def set_shaping_scale(self, scale: float):
        """
        This method allows a callback to update the annealing scale for shaped rewards.
        """
        self.shaping_scale = scale
