import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import itertools
from sklearn.preprocessing import OneHotEncoder

from sapai import Player, Pet, Food, Battle
from sapai.data import data

class SuperAutoPetsEnv(gym.Env):
    metadata = {'render_modes': ['human']}
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
    # Max turn limit to prevent infinite loops
    MAX_TURN = 25
    BAD_ACTION_PENALTY = -0.1

    # Max number of pets that can be on a team
    MAX_TEAM_PETS = 5
    # Max number of pets that can be in a shop
    MAX_SHOP_PETS = 6
    # Max number of foods that can be in a shop
    MAX_SHOP_FOODS = 3
    ALL_PETS = ["pet-ant", "pet-beaver", "pet-beetle", "pet-bluebird", "pet-cricket", "pet-duck", "pet-fish", "pet-horse", "pet-ladybug", "pet-mosquito", "pet-otter", "pet-pig", "pet-sloth", "pet-bat", "pet-crab", "pet-dodo", "pet-dog", "pet-dromedary", "pet-elephant", "pet-flamingo", "pet-hedgehog", "pet-peacock", "pet-rat", "pet-shrimp", "pet-spider", "pet-swan", "pet-tabby-cat", "pet-badger", "pet-blowfish", "pet-caterpillar", "pet-camel", "pet-hatching-chick", "pet-giraffe", "pet-kangaroo", "pet-owl", "pet-ox", "pet-puppy", "pet-rabbit", "pet-sheep", "pet-snail", "pet-tropical-fish", "pet-turtle", "pet-whale", "pet-bison", "pet-buffalo", "pet-deer", "pet-dolphin", "pet-hippo", "pet-llama", "pet-lobster", "pet-monkey", "pet-penguin", "pet-poodle", "pet-rooster", "pet-skunk", "pet-squirrel", "pet-worm", "pet-chicken", "pet-cow", "pet-crocodile", "pet-eagle", "pet-goat", "pet-microbe", "pet-parrot", "pet-rhino", "pet-scorpion", "pet-seal", "pet-shark", "pet-turkey", "pet-cat", "pet-boar", "pet-dragon", "pet-fly", "pet-gorilla", "pet-leopard", "pet-mammoth", "pet-octopus", "pet-sauropod", "pet-snake", "pet-tiger", "pet-tyrannosaurus", "pet-zombie-cricket", "pet-bus", "pet-zombie-fly", "pet-dirty-rat", "pet-chick", "pet-ram", "pet-butterfly", "pet-bee"]
    ALL_FOODS = ["food-apple", "food-honey", "food-cupcake", "food-meat-bone", "food-sleeping-pill", "food-garlic", "food-salad-bowl", "food-canned-food", "food-pear", "food-chili", "food-chocolate", "food-sushi", "food-melon", "food-mushroom", "food-pizza", "food-steak", "food-milk"]
    ALL_STATUSES = ["status-weak", "status-coconut-shield", "status-honey-bee", "status-bone-attack", "status-garlic-armor", "status-splash-attack", "status-melon-armor", "status-extra-life", "status-steak-attack", "status-poison-attack"]

    def __init__(self, opponent_generator, valid_actions_only=True, manual_battles=False):
        """
        Create a gym environment for Super Auto Pets supporting self-play between two agents.
        :param opponent_generator: Function to generate opponents (not used in self-play).
        :param valid_actions_only: If True, invalid actions will raise exceptions.
        :param manual_battles: If True, battles are controlled manually.
        """
        super(SuperAutoPetsEnv, self).__init__()

        # Validate parameters
        if manual_battles:
            assert opponent_generator is None
        else:
            assert opponent_generator is not None

        # Define action space: Tuple of two discrete actions (one for each agent)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.MAX_ACTIONS),  # Action for Agent 1
            spaces.Discrete(self.MAX_ACTIONS)   # Action for Agent 2
        ))

        # Define observation space: Concatenated observations for both agents
        len_obs_space = (len(self.ALL_PETS) + 2 + len(self.ALL_STATUSES)) * 11 + (len(self.ALL_FOODS) + 1) * 2 + 5
        self.observation_space = spaces.Box(low=0, high=255, shape=(len_obs_space * 2,), dtype=np.uint8)
        self.reward_range = (-1.0, 1.0)

        self.opponent_generator = opponent_generator
        self.valid_actions_only = valid_actions_only
        self.manual_battles = manual_battles

        # Initialize two players, shops, and teams
        self.player1 = Player()
        self.player2 = Player()

        self.shop1 = Shop(turn=1, pack="StandardPack")
        self.shop2 = Shop(turn=1, pack="StandardPack")

        self.player1.shop = self.shop1
        self.player2.shop = self.shop2

        self.team1 = Team(pack="StandardPack")
        self.team2 = Team(pack="StandardPack")

        self.player1.team = self.team1
        self.player2.team = self.team2

        # Initialize game state
        self.bad_action_reward_sum1 = 0
        self.bad_action_reward_sum2 = 0
        self.done = False
        self.info = {}
        self.last_actions = [None, None]

        self.rs_random = np.random.RandomState(None)
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to start a new game.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rs_random = np.random.RandomState(seed)
        else:
            self.rs_random = np.random.RandomState()

        # Reset players
        seed_state1 = self.rs_random.get_state()
        seed_state2 = self.rs_random.get_state()

        self.player1 = Player(seed_state=seed_state1)
        self.player2 = Player(seed_state=seed_state2)

        # Reset shops
        self.shop1 = Shop(turn=1, pack="StandardPack")
        self.shop2 = Shop(turn=1, pack="StandardPack")
        self.player1.shop = self.shop1
        self.player2.shop = self.shop2

        # Reset teams
        self.team1 = Team(pack="StandardPack")
        self.team2 = Team(pack="StandardPack")
        self.player1.team = self.team1
        self.player2.team = self.team2

        # Perform initial shop roll
        self.shop1.roll()
        self.shop2.roll()

        # Reset game state
        self.turn = 1
        self.done = False
        self.info = {}
        self.bad_action_reward_sum1 = 0
        self.bad_action_reward_sum2 = 0
        self.last_actions = [None, None]

        # Generate initial observations
        obs = self._encode_state()
        return obs, self.info  # Return info dict as per Gymnasium API

    def step(self, actions):
        """
        Execute one step in the environment with actions from both agents.
        :param actions: Tuple containing actions for Agent 1 and Agent 2.
        :return: Observation, rewards, done flag, truncated flag, and info dictionary.
        """
        if self.done:
            obs, info = self.reset()
            return obs, (0.0, 0.0), self.done, False, info

        action1, action2 = actions  # Unpack actions for both agents

        # Resolve actions
        self.resolve_action(agent_idx=0, action=action1)
        self.resolve_action(agent_idx=1, action=action2)

        # Execute battle phase if both agents have ended their turn
        if self.player1.turn_ended and self.player2.turn_ended and not self.manual_battles:
            battle_result = Battle(self.team1, self.team2).battle()
            self._player_fight_outcome(battle_result)
            self.player1.start_turn()
            self.player2.start_turn()
            self.turn += 1

            # Check for game termination
            self.done = self.is_done()

        # Assign rewards
        reward1, reward2 = self.get_reward()

        # Generate observations
        obs = self._encode_state()

        # Reset bad action penalties
        self.bad_action_reward_sum1 = 0
        self.bad_action_reward_sum2 = 0

        return obs, (reward1, reward2), self.done, False, self.info

    def resolve_action(self, agent_idx: int, action: int):
        """
        Resolve the action for a specific agent.
        :param agent_idx: Index of the agent (0 or 1).
        :param action: Action to be executed.
        """
        if not isinstance(action, int):
            action = action.item()

        player = self.player1 if agent_idx == 0 else self.player2
        shop = self.shop1 if agent_idx == 0 else self.shop2
        team = self.team1 if agent_idx == 0 else self.team2
        bad_action_sum = 'bad_action_reward_sum1' if agent_idx == 0 else 'bad_action_reward_sum2'

        if not self._is_valid_action(action):
            if self.valid_actions_only:
                raise RuntimeError(f"Agent {agent_idx+1} attempted invalid action {action}.")
            setattr(self, bad_action_sum, getattr(self, bad_action_sum) + self.BAD_ACTION_PENALTY)
            return

        # Resolve action
        action_to_play = self._avail_actions(agent_idx)[action]
        action_name = self._get_action_name(action_to_play).split(".")[-1]
        action_method = getattr(player, action_name)
        action_method(*action_to_play[1:])

        # Track last actions
        self.last_actions[agent_idx] = action

    @property
    def just_reordered(self):
        return any(get_action_name(a) == "reorder" for a in self.last_actions if a is not None)

    def render(self):
        """
        Render the current state of both players.
        """
        print("Player 1:")
        print(self.player1)
        print("Shop 1:")
        print(self.shop1)
        print("\nPlayer 2:")
        print(self.player2)
        print("Shop 2:")
        print(self.shop2)
        print(f"Turn: {self.turn}")
        print(f"Game Over: {self.done}")

    def is_done(self):
        """
        Determine if the game has ended.
        :return: Boolean indicating if the game is done.
        """
        # Example termination condition: one player's lives reach zero or max turn reached
        return (self.player1.lives <= 0 or self.player2.lives <= 0 or self.turn >= self.MAX_TURN)

    def get_reward(self):
        """
        Calculate rewards for both agents based on battle outcomes.
        :return: Tuple of rewards for Agent 1 and Agent 2.
        """
        reward1 = self.player1.wins / 10 + self.bad_action_reward_sum1
        reward2 = self.player2.wins / 10 + self.bad_action_reward_sum2
        return reward1, reward2

    def close(self):
        pass

    def _avail_end_turn(self, agent_idx):
        action_dict = {}
        action_num = self.ACTION_BASE_NUM["end_turn"]
        action_dict[action_num] = (self.player1.end_turn,) if agent_idx == 0 else (self.player2.end_turn,)
        return action_dict

    def _avail_buy_pets(self, agent_idx):
        action_dict = {}
        player = self.player1 if agent_idx == 0 else self.player2
        shop = self.shop1 if agent_idx == 0 else self.shop2
        team = self.team1 if agent_idx == 0 else self.team2

        if len(team) == self.MAX_TEAM_PETS:
            return action_dict

        pet_index = 0
        for shop_idx, shop_slot in enumerate(shop):
            if shop_slot.slot_type == "pet" and shop_slot.cost <= player.gold:
                action_num = self.ACTION_BASE_NUM["buy_pet"] + pet_index
                action_dict[action_num] = (player.buy_pet, shop_idx)
            pet_index += 1
        return action_dict

    def _avail_buy_foods(self, agent_idx):
        action_dict = {}
        player = self.player1 if agent_idx == 0 else self.player2
        shop = self.shop1 if agent_idx == 0 else self.shop2
        team = self.team1 if agent_idx == 0 else self.team2

        if len(team) == 0:
            return action_dict

        food_index = 0
        for shop_idx, shop_slot in enumerate(shop):
            if shop_slot.slot_type == "food" and shop_slot.cost <= player.gold:
                food_effect = data["foods"][shop_slot.obj.name]["ability"]["effect"]
                if shop_slot.obj.name in ["food-canned-food"] or \
                   ("target" in food_effect and food_effect["target"].get("kind") == "RandomFriend"):
                    action_num = self.ACTION_BASE_NUM["buy_food_team"] + food_index
                    action_dict[action_num] = (player.buy_food, shop_idx)
                else:
                    for team_idx, team_slot in enumerate(team):
                        if not team_slot.empty:
                            action_num = self.ACTION_BASE_NUM["buy_food"] + (food_index * self.MAX_TEAM_PETS) + team_idx
                            action_dict[action_num] = (player.buy_food, shop_idx, team_idx)
            food_index += 1
        return action_dict

    def _avail_buy_combine(self, agent_idx):
        action_dict = {}
        player = self.player1 if agent_idx == 0 else self.player2
        shop = self.shop1 if agent_idx == 0 else self.shop2
        team = self.team1 if agent_idx == 0 else self.team2

        team_names = {}
        for team_idx, slot in enumerate(team):
            if slot.empty:
                continue
            team_names.setdefault(slot.pet.name, []).append(team_idx)

        shop_pet_index = 0
        for shop_idx, shop_slot in enumerate(shop):
            if shop_slot.slot_type == "pet" and shop_slot.obj.name in team_names and shop_slot.cost <= player.gold:
                for team_idx in team_names[shop_slot.obj.name]:
                    action_num = self.ACTION_BASE_NUM["buy_combine"] + (shop_pet_index * self.MAX_TEAM_PETS) + team_idx
                    action_dict[action_num] = (player.buy_combine, shop_idx, team_idx)
            shop_pet_index += 1
        return action_dict

    def _avail_team_combine(self, agent_idx):
        action_dict = {}
        player = self.player1 if agent_idx == 0 else self.player2
        team = self.team1 if agent_idx == 0 else self.team2

        if len(team) <= 1:
            return action_dict

        team_names = {}
        for slot_idx, slot in enumerate(team):
            if slot.empty:
                continue
            team_names.setdefault(slot.pet.name, []).append(slot_idx)

        for key, value in team_names.items():
            if len(value) < 2:
                continue
            for idx0, idx1 in itertools.combinations(value, r=2):
                indexes = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
                action_num = self.ACTION_BASE_NUM["combine"] + indexes.index((idx0, idx1))
                action_dict[action_num] = (player.combine, idx0, idx1)
        return action_dict

    def _avail_sell(self, agent_idx):
        action_dict = {}
        player = self.player1 if agent_idx == 0 else self.player2
        team = self.team1 if agent_idx == 0 else self.team2

        if len([slot for slot in team if not slot.empty]) >= 5:
            for team_idx, slot in enumerate(team):
                if slot.empty:
                    continue
                action_num = self.ACTION_BASE_NUM["sell"] + team_idx
                action_dict[action_num] = (player.sell, team_idx)
        return action_dict

    def _avail_roll(self, agent_idx):
        action_dict = {}
        player = self.player1 if agent_idx == 0 else self.player2
        if player.gold > 1:
            action_dict[self.ACTION_BASE_NUM["roll"]] = (player.roll,)
        return action_dict

    def _avail_reorder(self, agent_idx):
        action_dict = {}
        player = self.player1 if agent_idx == 0 else self.player2

        if any(get_action_name(a) == "reorder" for a in self.last_actions if a is not None):
            return action_dict

        team = self.team1 if agent_idx == 0 else self.team2
        team_size = len(team)
        perms = itertools.permutations(range(team_size))
        next(perms)  # Skip do-nothing permutation

        for k, perm in enumerate(perms):
            action_num = self.ACTION_BASE_NUM["reorder"] + k
            action_dict[action_num] = (player.reorder, perm)
        return action_dict

    def _get_action_name(self, input_action):
        return str(input_action[0].__name__)

    def _avail_actions(self, agent_idx):
        """
        Get available actions for a specific agent.
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

        return all_avail_actions

    def _is_valid_action(self, action: int) -> bool:
        return action in range(self.MAX_ACTIONS)

    def action_masks(self) -> np.ndarray:
        """
        Returns action masks for both agents.
        """
        masks1 = np.zeros(self.MAX_ACTIONS, dtype=bool)
        masks2 = np.zeros(self.MAX_ACTIONS, dtype=bool)

        avail_actions1 = self._avail_actions(agent_idx=0).keys()
        avail_actions2 = self._avail_actions(agent_idx=1).keys()

        masks1[list(avail_actions1)] = True
        masks2[list(avail_actions2)] = True

        return np.concatenate([masks1, masks2])

    def _player_fight_outcome(self, outcome: int, agent_idx: int):
        """
        Assign fight outcomes to players based on battle results.
        """
        player = self.player1 if agent_idx == 0 else self.player2
        if outcome == 0:
            player.lf_winner = True
            player.wins += 1
        elif outcome == 2:
            player.lf_winner = False
        elif outcome == 1:
            player.lf_winner = False
            if player.turn <= 2:
                player.lives -= 1
            elif player.turn <= 4:
                player.lives -= 2
            else:
                player.lives -= 3
            player.lives = max(player.lives, 0)

    def _encode_pets(self, pets):
        arrays_to_concat = []
        for pet in pets:
            if pet.name == "pet-none":
                arrays_to_concat.append(np.zeros(len(self.ALL_PETS)))
                arrays_to_concat.append(np.zeros(2))
                arrays_to_concat.append(np.zeros(len(self.ALL_STATUSES)))
            else:
                arrays_to_concat.append(self._encode_single(pet.name, self.ALL_PETS))
                arrays_to_concat.append(np.array([pet.attack / 50, pet.health / 50]))
                if pet.status == "none":
                    arrays_to_concat.append(np.zeros(len(self.ALL_STATUSES)))
                else:
                    arrays_to_concat.append(self._encode_single(pet.status, self.ALL_STATUSES))
        return arrays_to_concat

    def _encode_foods(self, foods):
        arrays_to_concat = []
        for food_tuple in foods:
            (food, cost) = food_tuple
            if food.name == "food-none":
                arrays_to_concat.append(np.zeros(len(self.ALL_FOODS)))
                arrays_to_concat.append(np.zeros(1))
            else:
                arrays_to_concat.append(self._encode_single(food.name, self.ALL_FOODS))
                arrays_to_concat.append(np.array([cost / 3]))
        return arrays_to_concat

    def _get_shop_foods(self, agent_idx):
        shop = self.shop1 if agent_idx == 0 else self.shop2
        food_slots = []
        for slot in shop._slots:
            if slot.slot_type == "food":
                food_slots.append((slot.obj, slot.cost))
        return food_slots

    def _encode_state(self):
        """
        Encode the state for both agents, including their own state and opponent's previous board state.
        """
        # Encode Player 1's state
        encoded_team_pets1 = self._encode_pets([p.pet for p in self.team1])
        shop_pets1 = self.player1.shop.pets
        shop_foods1 = self._get_shop_foods(agent_idx=0)

        while len(shop_pets1) < self.MAX_SHOP_PETS:
            shop_pets1.append(Pet("pet-none"))
        while len(shop_foods1) < self.MAX_SHOP_FOODS:
            shop_foods1.append((Food("food-none"), 0))

        encoded_shop_pets1 = self._encode_pets(shop_pets1)
        encoded_shop_foods1 = self._encode_foods(shop_foods1)

        other_stats1 = np.array([
            self.player1.wins / 10,
            self.player1.lives / 10,
            min(self.player1.gold, 20) / 20,
            min(self.player1.turn, 25) / 25,
            min(self.player1.shop.shop_attack, 20) / 20
        ])

        state1 = np.concatenate(encoded_team_pets1 + encoded_shop_pets1 + encoded_shop_foods1 + [other_stats1])

        # Encode Player 2's state
        encoded_team_pets2 = self._encode_pets([p.pet for p in self.team2])
        shop_pets2 = self.player2.shop.pets
        shop_foods2 = self._get_shop_foods(agent_idx=1)

        while len(shop_pets2) < self.MAX_SHOP_PETS:
            shop_pets2.append(Pet("pet-none"))
        while len(shop_foods2) < self.MAX_SHOP_FOODS:
            shop_foods2.append((Food("food-none"), 0))

        encoded_shop_pets2 = self._encode_pets(shop_pets2)
        encoded_shop_foods2 = self._encode_foods(shop_foods2)

        other_stats2 = np.array([
            self.player2.wins / 10,
            self.player2.lives / 10,
            min(self.player2.gold, 20) / 20,
            min(self.player2.turn, 25) / 25,
            min(self.player2.shop.shop_attack, 20) / 20
        ])

        state2 = np.concatenate(encoded_team_pets2 + encoded_shop_pets2 + encoded_shop_foods2 + [other_stats2])

        # Concatenate both states
        combined_state = np.concatenate([state1, state2])

        # Normalize and convert to uint8
        combined_state = (combined_state * 255).clip(0, 255).astype(np.uint8)

        return combined_state

    @staticmethod
    def _encode_single(value, category):
        """
        One-hot encode a single categorical value.
        """
        np_array = np.array([[value]])
        encoder = OneHotEncoder(categories=[category], sparse_output=False, handle_unknown='ignore')
        onehot_encoded = encoder.fit_transform(np_array)
        collapsed = np.sum(onehot_encoded, axis=0)
        return collapsed

    def _player_fight_outcome(self, outcome: int):
        """
        Assign fight outcomes to both players based on battle results.
        :param outcome: Result of the battle (0: Player1 wins, 1: Player2 wins, 2: Draw)
        """
        if outcome == 0:
            self._player_fight_outcome_helper(0, outcome)
        elif outcome == 1:
            self._player_fight_outcome_helper(1, outcome)
        elif outcome == 2:
            # Handle draw if necessary
            pass

    def _player_fight_outcome_helper(self, agent_idx: int, outcome: int):
        """
        Helper function to assign fight outcomes.
        """
        player = self.player1 if agent_idx == 0 else self.player2
        if outcome == 0:
            player.lf_winner = True
            player.wins += 1
        elif outcome == 1:
            player.lf_winner = False
            if player.turn <= 2:
                player.lives -= 1
            elif player.turn <= 4:
                player.lives -= 2
            else:
                player.lives -= 3
            player.lives = max(player.lives, 0)

    def close(self):
        pass

def get_action_name(k: int) -> str:
    """
    Maps an integer action to its corresponding action name.
    """
    name_val = list(SuperAutoPetsEnv.ACTION_BASE_NUM.items())
    assert k >= 0
    for (start_name, _), (end_name, end_val) in zip(name_val[:-1], name_val[1:]):
        if k < end_val:
            return start_name
    else:
        return end_name
