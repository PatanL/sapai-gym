"""Version-preserving decision contract. None is legacy, zero is a real mask.

No opponent access, policy calls, or RNG draws happen during expansion.
"""
from dataclasses import dataclass
from numbers import Integral

from .action_schema import ActionType, AutoregressiveAction, FlatActionCodec, MAX_SHOP_SLOTS

PRIMITIVE_SCHEMA = "primitive-autoregressive-v1"
JOINT_SCHEMA = "primitive-plus-end-freeze-v1-20260905"


@dataclass(frozen=True)
class JointAction:
    primitive: AutoregressiveAction
    end_freeze_mask: int | None = None

    def __post_init__(self):
        if not isinstance(self.primitive, AutoregressiveAction):
            raise ValueError('Expected an autoregressive primitive')
        values = (self.primitive.action_type, self.primitive.source, self.primitive.target)
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
            raise ValueError('Action fields must be integers')
        # Canonical codec round-trip also rejects unused/noncanonical fields.
        if FlatActionCodec.decode(FlatActionCodec.encode(self.primitive)) != self.primitive:
            raise ValueError('Noncanonical primitive action')
        object.__setattr__(self, 'primitive', AutoregressiveAction(*(int(x) for x in values)))
        mask = self.end_freeze_mask
        if mask is not None:
            if isinstance(mask, bool) or not isinstance(mask, Integral) or not 0 <= mask < (1 << MAX_SHOP_SLOTS):
                raise ValueError('Invalid freeze subset')
            if mask and self.primitive.action_type != ActionType.END_TURN:
                raise ValueError('Freeze subset requires END_TURN')
            object.__setattr__(self, 'end_freeze_mask', int(mask))

    @property
    def schema(self):
        return PRIMITIVE_SCHEMA if self.end_freeze_mask is None else JOINT_SCHEMA

    @property
    def action_type(self):
        return self.primitive.action_type

    @property
    def source(self):
        return self.primitive.source

    @property
    def target(self):
        return self.primitive.target

    def as_tuple(self):
        raise ValueError('JointAction requires versioned serialization; a primitive tuple would discard its mask')

    def to_dict(self):
        return {'schema': self.schema, 'primitive': list(self.primitive.as_tuple()),
                'end_freeze_mask': self.end_freeze_mask}

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != {'schema', 'primitive', 'end_freeze_mask'}:
            raise ValueError('Decision payload must explicitly declare its schema and mask')
        result = cls(AutoregressiveAction(*payload['primitive']), payload['end_freeze_mask'])
        if payload['schema'] != result.schema:
            raise ValueError('Decision schema and mask disagree')
        return result

    def expand(self, core, side):
        from .end_freeze import expand_end_freeze
        if side not in (0, 1):
            raise ValueError('Invalid player seat')
        if self.primitive.action_type == ActionType.END_TURN:
            return expand_end_freeze(core, side, self.end_freeze_mask or 0)
        if FlatActionCodec.encode(self.primitive) not in core._avail_actions(side):
            raise ValueError('Decision is not legal in this state')
        return (self.primitive,)


def step_decisions(env, decisions):
    """Validate both seats before applying any part of either decision."""
    from .end_freeze import step_with_end_freeze
    if len(decisions) != 2 or any(d is not None and not isinstance(d, JointAction) for d in decisions):
        raise ValueError('Expected two decisions, with None only for inactive seats')
    return step_with_end_freeze(
        env, *(d.primitive if d is not None else None for d in decisions),
        mask_ids=tuple(0 if d is None or d.end_freeze_mask is None else d.end_freeze_mask for d in decisions),
    )
