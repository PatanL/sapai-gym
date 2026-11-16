"""Explicit migration helpers for the opt-in joint end-freeze action schema."""

import math


from .joint_action import JOINT_SCHEMA as END_FREEZE_SCHEMA, PRIMITIVE_SCHEMA
HEAD_PREFIX = 'end_freeze_head.'


def initialize_end_freeze_from_primitive(policy, primitive_state):
    """Preserve every parent weight, initializing ONLY the declared new head."""
    if policy.end_freeze_head is None:
        raise ValueError('Migration target must enable end-freeze')
    target = policy.state_dict()
    parent_keys = {key for key in target if not key.startswith(HEAD_PREFIX)}
    if set(primitive_state) != parent_keys:
        raise ValueError('End-freeze migration requires an exact primitive parent architecture')
    target.update(primitive_state)
    return policy.load_state_dict(target)


def attach_end_freeze_optimizer_group(policy, optimizer, learning_rate):
    """Append fresh head parameters after loading legacy optimizer groups.

    The caller constructs the original groups excluding this head, then loads
    their original state normally. No positional guessing or zeroing of parent
    Adam moments is performed here.
    """
    if policy.end_freeze_head is None:
        raise ValueError('Policy does not enable end-freeze')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('End-freeze learning rate must be finite and positive')
    parameters = list(policy.end_freeze_head.parameters())
    existing = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    if any(id(parameter) in existing for parameter in parameters):
        raise ValueError('End-freeze parameters already belong to the optimizer')
    optimizer.add_param_group({'params': parameters, 'lr': learning_rate, 'name': 'end_freeze'})
