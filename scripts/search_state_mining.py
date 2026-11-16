"""Sample decisions across a real shop sequence without prescribing actions."""

from generate_search_demonstrations import opportunity_reasons, policy_action, step_policy_decisions
from sapai_gym1.rl.joint_action import JointAction
from sapai_gym1.rl import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import ActionType, AutoregressiveAction, FlatActionCodec
from collections import Counter


MINING_VERSION = "uniform-decision-gold-strata-v1-20260905"
GOLD_BANDS = ("zero", "low", "funded", "any")
OPENING_SELECTION_VERSION = 'opening-freeze-opportunities-v1-20260905'


def act(policy, observation, memory, device):
    return policy_action(policy, observation, memory, device, False)


def opening_freeze_reasons(core, side):
    """Identify available copy-preservation choices, without labeling a move."""
    player = core.agent if side == 0 else core.opponent
    if core.turn != 1 or player.gold != 0 or player.turn_ended:
        return ()
    copies = Counter()
    for slot in player.team:
        if not slot.empty and slot.pet.level == 1:
            copies[slot.pet.name] += int(slot.pet.total_experience) + 1
    shop_copies = Counter(slot.obj.name for slot in player.shop
                          if not slot.empty and slot.slot_type in {'pet', 'levelup'})
    legal = core._avail_actions(side)
    available_names = {
        slot.obj.name for index, slot in enumerate(player.shop)
        if not slot.empty and slot.slot_type in {'pet', 'levelup'} and not slot.frozen
        and FlatActionCodec.encode(AutoregressiveAction(ActionType.FREEZE, index)) in legal
    }
    reasons = []
    if any(copies[name] >= 2 for name in available_names):
        reasons.append('opening-levelup-copy-available')
    if any(copies[name] == 1 and shop_copies[name] >= 2 for name in available_names):
        reasons.append('opening-two-shop-copies-available')
    return tuple(reasons)


def matches_gold_band(gold, band):
    if band == "any":
        return True
    if band == "zero":
        return gold == 0
    if band == "low":
        return 0 < gold < 3
    if band == "funded":
        return gold >= 3
    raise ValueError(f"Unknown gold band {band}")


def mine_match_opportunity(
    policy, rng, device, minimum_turn, maximum_turn, band="any", *, with_replay=False,
    opponent_policy=None, acting_side=None,
    opportunity_filter=None,
):
    """Reservoir-sample one eligible decision uniformly within a sampled game.

    Return its full environment only to the evaluator. Callers must construct
    PublicSearchState before passing it to a planner. Snapshot memories belong
    to the instant before the selected action, not the end of the mined game.
    """
    if band not in GOLD_BANDS or not 1 <= minimum_turn <= maximum_turn:
        raise ValueError("Invalid mining band or turn window")
    if acting_side not in (None, 0, 1):
        raise ValueError("Invalid acting side")
    if opponent_policy is not None and acting_side is None:
        raise ValueError("A historical opponent requires an explicit acting side")
    policies = [policy, policy]
    if opponent_policy is not None:
        policies[1 - acting_side] = opponent_policy
    env = ParallelSelfPlayEnv()
    reset_seed = int(rng.integers(0, 2**31 - 1))
    observations = env.reset(seed=reset_seed)
    memories = [network.initial_memory(1, device="cpu")[0] for network in policies]
    selected = None
    eligible = 0
    replay_actions = []
    while env.core.turn <= maximum_turn:
        active = env.active_players()
        if env.core.turn >= minimum_turn:
            for side in range(2):
                if acting_side is not None and side != acting_side:
                    continue
                player = env.core.agent if side == 0 else env.core.opponent
                if not active[side] or not matches_gold_band(player.gold, band):
                    continue
                reasons = (opportunity_filter(env.core, side) if opportunity_filter is not None
                           else opportunity_reasons(env.core, side))
                if not reasons:
                    continue
                eligible += 1
                if int(rng.integers(eligible)) == 0:
                    selected = (
                        env.clone(),
                        [memory.clone() for memory in memories],
                        side,
                        tuple(reasons),
                    )
                    if with_replay:
                        selected += ({
                            "resetSeed": reset_seed,
                            "actions": list(replay_actions),
                        },)
        actions = [None, None]
        for side in range(2):
            if active[side]:
                actions[side], memories[side] = act(
                    policies[side], observations[side], memories[side], device
                )
        step = step_policy_decisions(env, actions)
        if with_replay:
            replay_actions.append([
                None if action is None else (action if isinstance(action, JointAction) else JointAction(action)).to_dict()
                for action in actions
            ])
        observations = step.observations
        if step.terminated:
            break
    return selected
