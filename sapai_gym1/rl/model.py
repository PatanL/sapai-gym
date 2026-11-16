"""Recurrent entity-transformer policy and autoregressive action decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .end_freeze import EndFreezeHead

from .action_schema import (
    ActionType,
    MAX_SHOP_SLOTS,
    MAX_TEAM_SLOTS,
    NO_SOURCE,
    NO_TARGET,
    NUM_ACTION_TYPES,
    NUM_SOURCES,
    NUM_TARGETS,
)
from .entity_observation import (
    EntityKind,
    NUM_ENTITIES,
    NUM_NUMERIC_FEATURES,
    OPPONENT_HISTORY_BOARDS,
    OPPONENT_HISTORY_NUMERIC_FEATURES,
    OPPONENT_HISTORY_SLOTS,
)


NUM_LEGALITY_FEATURES = (
    NUM_ACTION_TYPES
    + NUM_ACTION_TYPES * NUM_SOURCES
    + NUM_ACTION_TYPES * NUM_SOURCES * NUM_TARGETS
)


@dataclass
class EntityTransformerConfig:
    num_items: int
    num_statuses: int
    d_model: int = 384
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1536
    memory_dim: int = 512
    dropout: float = 0.0
    opponent_history_dim: int = 128
    opponent_history_layers: int = 1
    end_freeze: bool = False
    # Objective-only ablation: does not change weights, actions or likelihoods.
    entropy_mode: str = "legacy_path"

    def __post_init__(self):
        if self.entropy_mode not in ("legacy_path", "exact_joint"):
            raise ValueError("Unknown entropy mode")


@dataclass(frozen=True)
class EncodedDecision:
    observation: dict[str, torch.Tensor]
    tokens: torch.Tensor
    context: torch.Tensor
    memory: torch.Tensor
    opponent_prediction: dict[str, torch.Tensor] | None
    history: torch.Tensor


@dataclass
class ActionOnlyPolicyOutput:
    actions: torch.Tensor
    log_prob: torch.Tensor
    memory: torch.Tensor
    end_freeze_mask: torch.Tensor | None = None


@dataclass
class PolicyOutput:
    actions: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    memory: torch.Tensor
    type_logits: torch.Tensor
    source_logits: torch.Tensor
    target_logits: torch.Tensor
    battle_logits: torch.Tensor
    board_strength: torch.Tensor
    opponent_pet_logits: torch.Tensor
    opponent_status_logits: torch.Tensor
    opponent_level_logits: torch.Tensor
    opponent_stats: torch.Tensor
    opponent_temporary_logits: torch.Tensor
    end_freeze_mask: torch.Tensor | None = None


@dataclass
class AllActionPolicyOutput:
    joint_log_prob: torch.Tensor
    value: torch.Tensor
    memory: torch.Tensor
    opponent_pet_logits: torch.Tensor
    opponent_status_logits: torch.Tensor
    opponent_level_logits: torch.Tensor
    opponent_stats: torch.Tensor
    opponent_temporary_logits: torch.Tensor


@dataclass
class SequencePolicyOutput:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    memory_states: torch.Tensor
    final_memory: torch.Tensor
    battle_logits: torch.Tensor
    board_strength: torch.Tensor
    opponent_pet_logits: torch.Tensor
    opponent_status_logits: torch.Tensor
    opponent_level_logits: torch.Tensor
    opponent_stats: torch.Tensor
    opponent_temporary_logits: torch.Tensor


class EntityTransformerPolicy(nn.Module):
    """Encode game entities and decode type -> source -> target."""

    SHOP_SOURCE_TYPES = {
        int(ActionType.BUY_PET),
        int(ActionType.BUY_FOOD),
        int(ActionType.FREEZE),
        int(ActionType.UNFREEZE),
    }

    def __init__(self, config: EntityTransformerConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.item_embedding = nn.Embedding(config.num_items, d_model)
        self.kind_embedding = nn.Embedding(len(EntityKind), d_model)
        self.status_embedding = nn.Embedding(config.num_statuses, d_model)
        self.position_embedding = nn.Embedding(MAX_SHOP_SLOTS + 1, d_model)
        self.numeric_projection = nn.Sequential(
            nn.Linear(NUM_NUMERIC_FEATURES, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Value/global context needs exact legality, not only masks applied to
        # logits. Zero initialization keeps pre-v18 checkpoints neutral until
        # the new branch learns from these features.
        self.legality_projection = nn.Linear(
            NUM_LEGALITY_FEATURES, d_model, bias=False
        )
        history_dim = config.opponent_history_dim
        self.opponent_history_item_embedding = nn.Embedding(
            config.num_items, history_dim
        )
        self.opponent_history_status_embedding = nn.Embedding(
            config.num_statuses, history_dim
        )
        self.opponent_history_position_embedding = nn.Embedding(
            MAX_TEAM_SLOTS, history_dim
        )
        self.opponent_history_numeric_projection = nn.Sequential(
            nn.Linear(OPPONENT_HISTORY_NUMERIC_FEATURES, history_dim),
            nn.GELU(),
            nn.Linear(history_dim, history_dim),
        )
        self.opponent_history_board_projection = nn.Linear(
            OPPONENT_HISTORY_BOARDS, history_dim
        )
        self.opponent_history_cls = nn.Parameter(torch.empty(history_dim))
        history_layer = nn.TransformerEncoderLayer(
            d_model=history_dim,
            nhead=4,
            dim_feedforward=history_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.opponent_history_transformer = nn.TransformerEncoder(
            history_layer,
            num_layers=config.opponent_history_layers,
            norm=nn.LayerNorm(history_dim),
        )
        self.opponent_belief_adapter = nn.Linear(
            history_dim, d_model, bias=False
        )
        self.opponent_pet_head = nn.Linear(
            history_dim, OPPONENT_HISTORY_SLOTS * config.num_items
        )
        self.opponent_status_head = nn.Linear(
            history_dim, OPPONENT_HISTORY_SLOTS * config.num_statuses
        )
        self.opponent_level_head = nn.Linear(
            history_dim, OPPONENT_HISTORY_SLOTS * 4
        )
        self.opponent_stats_head = nn.Linear(
            history_dim, OPPONENT_HISTORY_SLOTS * 2
        )
        self.opponent_temporary_head = nn.Linear(
            history_dim, OPPONENT_HISTORY_SLOTS * 2
        )
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, norm=nn.LayerNorm(d_model)
        )

        self.memory_cell = nn.GRUCell(d_model, config.memory_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(d_model + config.memory_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.action_type_embedding = nn.Embedding(NUM_ACTION_TYPES, d_model)
        self.no_source_embedding = nn.Parameter(torch.empty(d_model))
        self.no_target_embedding = nn.Parameter(torch.empty(d_model))

        self.type_head = nn.Linear(d_model, NUM_ACTION_TYPES)
        self.source_query = nn.Linear(d_model * 2, d_model)
        self.source_key = nn.Linear(d_model, d_model)
        self.target_query = nn.Linear(d_model * 3, d_model)
        self.target_key = nn.Linear(d_model, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.battle_head = nn.Linear(d_model, 3)
        self.board_strength_head = nn.Linear(d_model, 1)
        self.reset_parameters()
        # Appended after the legacy initialization so the disabled model and
        # common-parent parameters are unaffected by this opt-in action head.
        self.end_freeze_head = EndFreezeHead(d_model) if config.end_freeze else None

    def reset_parameters(self):
        nn.init.normal_(self.no_source_embedding, std=0.02)
        nn.init.normal_(self.no_target_embedding, std=0.02)
        nn.init.zeros_(self.legality_projection.weight)
        nn.init.normal_(self.opponent_history_cls, std=0.02)
        nn.init.zeros_(self.opponent_belief_adapter.weight)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load pre-v19 checkpoints with neutral additive adapters."""

        state_dict = dict(state_dict)
        current_state = super().state_dict()
        migration_prefixes = (
            "legality_projection.",
            "opponent_history_",
            "opponent_belief_adapter.",
            "opponent_pet_head.",
            "opponent_status_head.",
            "opponent_level_head.",
            "opponent_stats_head.",
            "opponent_temporary_head.",
        )
        for key, value in current_state.items():
            if key not in state_dict and key.startswith(migration_prefixes):
                state_dict[key] = value.detach().clone()
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

    def initial_memory(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.memory_dim, device=device)

    def forward(
        self,
        observation: dict[str, torch.Tensor],
        memory: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
        *,
        allow_end_freeze: bool = False,
        end_freeze_masks: torch.Tensor | None = None,
        entropy_mode_override: str | None = None,
    ) -> PolicyOutput:
        self._validate_action_schema(actions, allow_end_freeze, end_freeze_masks)
        features = self.encode_decision(observation, memory)
        return self.decode_decision(features, actions, deterministic,
                                    allow_end_freeze=allow_end_freeze,
                                    end_freeze_masks=end_freeze_masks,
                                    entropy_mode_override=entropy_mode_override)

    def _validate_action_schema(self, actions, allow_end_freeze, end_freeze_masks):
        if self.end_freeze_head is not None:
            if not allow_end_freeze:
                raise ValueError('This checkpoint requires an end-freeze-aware action consumer')
            if actions is not None and end_freeze_masks is None:
                raise ValueError('Recorded joint actions require their end-freeze masks')
        elif end_freeze_masks is not None:
            raise ValueError('Primitive policies cannot evaluate end-freeze masks')

    def encode_decision(self, observation, memory=None, *, forecast=True) -> EncodedDecision:
        """Return features and proposed next memory without mutating live memory."""
        tokens = self.encode_entities(observation)
        batch_size = tokens.shape[0]
        if memory is None:
            memory = self.initial_memory(batch_size, tokens.device)
        new_memory = self.memory_cell(tokens[:, 0], memory)
        context = self.context_projection(
            torch.cat([tokens[:, 0], new_memory], dim=-1)
        )
        opponent_belief = self.encode_opponent_history(observation)
        context = context + self.opponent_belief_adapter(opponent_belief)
        opponent_prediction = self.predict_opponent_board(opponent_belief) if forecast else None
        return EncodedDecision(observation, tokens, context, new_memory, opponent_prediction, opponent_belief)

    def forecast_opponent(self, features):
        return (features.opponent_prediction if features.opponent_prediction is not None
                else self.predict_opponent_board(features.history))

    def value_from_encoded(self, features):
        return self.value_head(features.context).squeeze(-1)

    def value_only(self, observation, memory=None):
        """Value evaluation without action decoding or unused auxiliary heads.

        Call under no_grad for bootstrap/search; this method does not secretly
        change the caller's train/eval or gradient mode.
        """
        return self.value_from_encoded(self.encode_decision(observation, memory, forecast=False))

    def sample_joint_action(self, features, *, deterministic=False):
        """Actor-only output, including the exact sampled joint likelihood."""
        return self.decode_decision(features, deterministic=deterministic,
                                    allow_end_freeze=self.config.end_freeze, action_only=True)

    def act(self, observation, memory=None, *, deterministic=False):
        return self.sample_joint_action(self.encode_decision(observation, memory, forecast=False),
                                        deterministic=deterministic)

    def decode_decision(self, features, actions=None, deterministic=False, *,
                        allow_end_freeze=False, end_freeze_masks=None, action_only=False,
                        entropy_mode_override=None):
        """Decode or rescore cached features; never execute another GRU update."""
        self._validate_action_schema(actions, allow_end_freeze, end_freeze_masks)
        entropy_mode = self.config.entropy_mode if entropy_mode_override is None else entropy_mode_override
        if entropy_mode not in ('legacy_path', 'exact_joint'):
            raise ValueError('Unknown requested entropy mode')
        observation, tokens, context = features.observation, features.tokens, features.context
        batch_size = tokens.shape[0]

        type_logits = self._masked_logits(
            self.type_head(context), observation["type_mask"]
        )
        type_dist = Categorical(logits=type_logits)
        action_type = self._select(type_dist, actions, 0, deterministic)

        source_entities = self._source_entities(tokens, action_type)
        type_embedding = self.action_type_embedding(action_type)
        source_query = self.source_query(
            torch.cat([context, type_embedding], dim=-1)
        )
        source_logits = torch.einsum(
            "bd,bsd->bs", source_query, self.source_key(source_entities)
        ) / math.sqrt(self.config.d_model)
        source_mask = self._conditional_mask(
            observation["source_mask"], action_type
        )
        source_logits = self._masked_logits(source_logits, source_mask)
        source_dist = Categorical(logits=source_logits)
        source = self._select(source_dist, actions, 1, deterministic)

        selected_source = source_entities.gather(
            1,
            source[:, None, None].expand(-1, 1, self.config.d_model),
        ).squeeze(1)
        target_entities = torch.cat(
            [
                tokens[:, 1 : 1 + MAX_TEAM_SLOTS],
                self.no_target_embedding[None, None].expand(batch_size, 1, -1),
            ],
            dim=1,
        )
        target_query = self.target_query(
            torch.cat([context, type_embedding, selected_source], dim=-1)
        )
        target_logits = torch.einsum(
            "bd,btd->bt", target_query, self.target_key(target_entities)
        ) / math.sqrt(self.config.d_model)
        target_mask = self._target_mask(
            observation["target_mask"], action_type, source
        )
        target_logits = self._masked_logits(target_logits, target_mask)
        target_dist = Categorical(logits=target_logits)
        target = self._select(target_dist, actions, 2, deterministic)

        chosen_actions = torch.stack([action_type, source, target], dim=-1)
        log_prob = (
            type_dist.log_prob(action_type)
            + source_dist.log_prob(source)
            + target_dist.log_prob(target)
        )
        if action_only:
            mask_ids = None
            if self.end_freeze_head is not None:
                start = 1 + MAX_TEAM_SLOTS
                freeze = self.end_freeze_head(
                    context, tokens[:, start:start + MAX_SHOP_SLOTS], observation,
                    action_type, mask_ids=end_freeze_masks, deterministic=deterministic,
                    compute_entropy=False,
                )
                mask_ids = freeze.mask_ids
                log_prob = log_prob + freeze.log_prob
            return ActionOnlyPolicyOutput(chosen_actions, log_prob, features.memory, mask_ids)
        entropy = (
            type_dist.entropy() + source_dist.entropy() + target_dist.entropy()
        )
        if entropy_mode == "exact_joint":
            entropy = self.exact_primitive_entropy(tokens, context, observation)
        end_freeze_mask = None
        if self.end_freeze_head is not None:
            freeze, conditional_entropy = self._end_freeze_decision(
                context, tokens, observation, type_dist, action_type,
                masks=end_freeze_masks, deterministic=deterministic,
            )
            end_freeze_mask = freeze.mask_ids
            log_prob = log_prob + freeze.log_prob
            entropy = entropy + conditional_entropy
        return PolicyOutput(
            actions=chosen_actions,
            log_prob=log_prob,
            entropy=entropy,
            value=self.value_head(context).squeeze(-1),
            memory=features.memory,
            type_logits=type_logits,
            source_logits=source_logits,
            target_logits=target_logits,
            battle_logits=self.battle_head(context),
            board_strength=self.board_strength_head(context).squeeze(-1),
            end_freeze_mask=end_freeze_mask,
            **self.forecast_opponent(features),
        )

    def _end_freeze_decision(self, context, tokens, observation, type_dist, action_types,
                             *, masks=None, deterministic=False):
        start = 1 + MAX_TEAM_SLOTS
        output = self.end_freeze_head(
            context, tokens[:, start:start + MAX_SHOP_SLOTS], observation,
            action_types, mask_ids=masks, deterministic=deterministic,
        )
        # H(mask | END,state) is weighted by P(END|state), NOT by whether the
        # sampled/recorded action happened to be END. This gives the exact
        # additional conditional-entropy term and its action-type gradient.
        end_types = torch.full_like(action_types, int(ActionType.END_TURN))
        distribution, _ = self.end_freeze_head.distribution(
            output.slot_logits, observation, end_types
        )
        entropy = type_dist.probs[:, ActionType.END_TURN].float() * distribution.entropy()
        return output, entropy

    def evaluate_all_actions(
        self,
        observation: dict[str, torch.Tensor],
        memory: torch.Tensor | None = None,
    ) -> AllActionPolicyOutput:
        """Score every autoregressive action after one entity encoding.

        The old search evaluator repeated the full transformer once for every
        legal action. Here the transformer and recurrent update run once per
        state; only the inexpensive type/source/target decoder heads are
        expanded over all conditional choices.
        """

        if self.end_freeze_head is not None:
            raise ValueError('Primitive action enumeration does not support end-freeze checkpoints')
        features = self.encode_decision(observation, memory)
        return AllActionPolicyOutput(
            joint_log_prob=self.primitive_log_probabilities(features),
            value=self.value_head(features.context).squeeze(-1),
            memory=features.memory,
            **self.forecast_opponent(features),
        )

    def primitive_log_probabilities(self, features):
        """Primitive MARGINALS, not full joint-decision likelihoods when enabled.

        Joint consumers must additionally condition END on its freeze subset.
        The primitive-only public entrypoint above deliberately remains guarded.
        """
        return self._primitive_log_probabilities(features.tokens, features.context,
                                                  features.observation)

    def _primitive_log_probabilities(self, tokens, context, observation):
        batch_size = tokens.shape[0]

        type_logits = self._masked_logits(
            self.type_head(context), observation["type_mask"]
        )
        type_log_prob = torch.log_softmax(type_logits, dim=-1)

        action_types = torch.arange(
            NUM_ACTION_TYPES, device=tokens.device
        )[None].expand(batch_size, -1)
        flat_types = action_types.reshape(-1)
        flat_tokens = tokens[:, None].expand(
            -1, NUM_ACTION_TYPES, -1, -1
        ).reshape(
            batch_size * NUM_ACTION_TYPES,
            tokens.shape[1],
            self.config.d_model,
        )
        source_entities = self._source_entities(
            flat_tokens, flat_types
        ).reshape(
            batch_size,
            NUM_ACTION_TYPES,
            NUM_SOURCES,
            self.config.d_model,
        )
        type_embedding = self.action_type_embedding(action_types)
        source_query = self.source_query(
            torch.cat(
                [
                    context[:, None].expand(-1, NUM_ACTION_TYPES, -1),
                    type_embedding,
                ],
                dim=-1,
            )
        )
        source_logits = torch.einsum(
            "btd,btsd->bts",
            source_query,
            self.source_key(source_entities),
        ) / math.sqrt(self.config.d_model)
        source_mask = observation["source_mask"].bool().clone()
        invalid_types = ~observation["type_mask"].bool()
        source_mask[..., NO_SOURCE] |= invalid_types
        source_logits = self._masked_logits(source_logits, source_mask)
        source_log_prob = torch.log_softmax(source_logits, dim=-1)

        target_entities = torch.cat(
            [
                tokens[:, 1 : 1 + MAX_TEAM_SLOTS],
                self.no_target_embedding[None, None].expand(
                    batch_size, 1, -1
                ),
            ],
            dim=1,
        )
        target_query = self.target_query(
            torch.cat(
                [
                    context[:, None, None].expand(
                        -1, NUM_ACTION_TYPES, NUM_SOURCES, -1
                    ),
                    type_embedding[:, :, None].expand(
                        -1, -1, NUM_SOURCES, -1
                    ),
                    source_entities,
                ],
                dim=-1,
            )
        )
        target_logits = torch.einsum(
            "btsd,bkd->btsk",
            target_query,
            self.target_key(target_entities),
        ) / math.sqrt(self.config.d_model)
        target_mask = observation["target_mask"].bool().clone()
        missing_targets = ~target_mask.any(dim=-1)
        target_mask[..., NO_TARGET] |= missing_targets
        target_logits = self._masked_logits(target_logits, target_mask)
        target_log_prob = torch.log_softmax(target_logits, dim=-1)
        joint_log_prob = (
            type_log_prob[:, :, None, None]
            + source_log_prob[:, :, :, None]
            + target_log_prob
        )
        # Explicit -inf on invalid complete paths prevents underflowed or
        # dummy conditional choices from being mistaken for legal actions.
        legal = (observation['type_mask'].bool()[:, :, None, None]
                 & observation['source_mask'].bool()[:, :, :, None]
                 & observation['target_mask'].bool())
        return joint_log_prob.masked_fill(~legal, -torch.inf)

    def exact_primitive_entropy(self, tokens, context, observation):
        """Exact chain-rule entropy and upstream probability-weight gradients.

        Enumerate only the small decoder, reusing the entity/context encoding.
        FP32 reduction and finite log replacement avoid 0 * -inf / NaN.
        """
        log_p = self._primitive_log_probabilities(tokens, context, observation).float()
        finite_log_p = log_p.masked_fill(~torch.isfinite(log_p), 0.)
        return -(log_p.exp() * finite_log_p).sum(dim=(1, 2, 3))

    def encode_opponent_history(self, observation):
        item_ids = observation["opponent_history_item_ids"].long()
        batch_size = item_ids.shape[0]
        tokens = (
            self.opponent_history_item_embedding(item_ids)
            + self.opponent_history_status_embedding(
                observation["opponent_history_status_ids"].long()
            )
            + self.opponent_history_position_embedding(
                observation["opponent_history_positions"].long()
            )
            + self.opponent_history_numeric_projection(
                observation["opponent_history_numeric"].float()
            )
        ).reshape(
            batch_size,
            OPPONENT_HISTORY_BOARDS * OPPONENT_HISTORY_SLOTS,
            self.config.opponent_history_dim,
        )
        cls = self.opponent_history_cls[None, None].expand(
            batch_size, 1, -1
        )
        cls = cls + self.opponent_history_board_projection(
            observation["opponent_history_board_mask"].float()
        )[:, None]
        tokens = torch.cat([cls, tokens], dim=1)
        padding = ~observation["opponent_history_mask"].bool().flatten(1)
        padding = torch.cat(
            [
                torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=padding.device
                ),
                padding,
            ],
            dim=1,
        )
        encoded = self.opponent_history_transformer(
            tokens, src_key_padding_mask=padding
        )
        return encoded[:, 0]

    def predict_opponent_board(self, belief):
        batch_size = belief.shape[0]
        return {
            "opponent_pet_logits": self.opponent_pet_head(belief).reshape(
                batch_size, OPPONENT_HISTORY_SLOTS, self.config.num_items
            ),
            "opponent_status_logits": self.opponent_status_head(
                belief
            ).reshape(
                batch_size,
                OPPONENT_HISTORY_SLOTS,
                self.config.num_statuses,
            ),
            "opponent_level_logits": self.opponent_level_head(belief).reshape(
                batch_size, OPPONENT_HISTORY_SLOTS, 4
            ),
            "opponent_stats": torch.sigmoid(
                self.opponent_stats_head(belief).reshape(
                    batch_size, OPPONENT_HISTORY_SLOTS, 2
                )
            ),
            "opponent_temporary_logits": self.opponent_temporary_head(
                belief
            ).reshape(batch_size, OPPONENT_HISTORY_SLOTS, 2),
        }

    def encode_entities(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        token_input = (
            self.item_embedding(observation["item_ids"].long())
            + self.kind_embedding(observation["kind_ids"].long())
            + self.status_embedding(observation["status_ids"].long())
            + self.position_embedding(observation["positions"].long())
            + self.numeric_projection(observation["numeric"].float())
        )
        legality = torch.cat(
            [
                observation["type_mask"].float().flatten(start_dim=1),
                observation["source_mask"].float().flatten(start_dim=1),
                observation["target_mask"].float().flatten(start_dim=1),
            ],
            dim=1,
        )
        token_input[:, 0] = (
            token_input[:, 0] + self.legality_projection(legality)
        )
        token_input = self.input_norm(token_input)
        padding_mask = ~observation["entity_mask"].bool()
        return self.transformer(
            token_input, src_key_padding_mask=padding_mask
        )

    def evaluate_sequence(
        self,
        observation: dict[str, torch.Tensor],
        initial_memory: torch.Tensor,
        actions: torch.Tensor,
        *,
        end_freeze_masks: torch.Tensor | None = None,
        forecast_first_only: bool = False,
        context_residual: torch.Tensor | None = None,
    ) -> SequencePolicyOutput:
        """Evaluate teacher actions for observation tensors shaped ``[T,B,...]``.

        Entity attention is independent across time, so it is evaluated as one
        large GPU batch.  Only the small recurrent memory update remains a
        chronological loop.
        """

        if self.end_freeze_head is not None:
            if end_freeze_masks is None or end_freeze_masks.shape != actions.shape[:2]:
                raise ValueError('Recorded joint sequences require end-freeze masks shaped [T,B]')
        elif end_freeze_masks is not None:
            raise ValueError('Primitive policies cannot evaluate end-freeze masks')
        time_steps, batch_size = actions.shape[:2]
        flat_observation = {
            key: value.reshape(time_steps * batch_size, *value.shape[2:])
            for key, value in observation.items()
        }
        flat_tokens = self.encode_entities(flat_observation)
        flat_opponent_belief = self.encode_opponent_history(flat_observation)
        forecast_belief = (flat_opponent_belief[:batch_size] if forecast_first_only else flat_opponent_belief)
        flat_opponent_prediction = self.predict_opponent_board(forecast_belief)
        tokens = flat_tokens.reshape(
            time_steps, batch_size, NUM_ENTITIES, self.config.d_model
        )

        memory = initial_memory
        memories = []
        contexts = []
        for time_index in range(time_steps):
            memory = self.memory_cell(tokens[time_index, :, 0], memory)
            memories.append(memory)
            contexts.append(
                self.context_projection(
                    torch.cat([tokens[time_index, :, 0], memory], dim=-1)
                )
                + self.opponent_belief_adapter(
                    flat_opponent_belief.reshape(
                        time_steps,
                        batch_size,
                        self.config.opponent_history_dim,
                    )[time_index]
                )
            )
        memory_states = torch.stack(memories)
        flat_context = torch.stack(contexts).reshape(
            time_steps * batch_size, self.config.d_model
        )
        if context_residual is not None:
            if context_residual.shape != (time_steps, batch_size, self.config.d_model):
                raise ValueError('Causal context residual must have shape [T,B,d_model]')
            if not torch.isfinite(context_residual).all():
                raise ValueError('Nonfinite causal context residual')
            flat_context = flat_context + context_residual.reshape_as(flat_context)
        flat_actions = actions.reshape(time_steps * batch_size, 3)

        type_logits = self._masked_logits(
            self.type_head(flat_context), flat_observation["type_mask"]
        )
        action_type = flat_actions[:, 0].long()
        type_dist = Categorical(logits=type_logits)

        source_entities = self._source_entities(flat_tokens, action_type)
        type_embedding = self.action_type_embedding(action_type)
        source_query = self.source_query(
            torch.cat([flat_context, type_embedding], dim=-1)
        )
        source_logits = torch.einsum(
            "bd,bsd->bs", source_query, self.source_key(source_entities)
        ) / math.sqrt(self.config.d_model)
        source_logits = self._masked_logits(
            source_logits,
            self._conditional_mask(
                flat_observation["source_mask"], action_type
            ),
        )
        source = flat_actions[:, 1].long()
        source_dist = Categorical(logits=source_logits)
        selected_source = source_entities.gather(
            1,
            source[:, None, None].expand(-1, 1, self.config.d_model),
        ).squeeze(1)

        target_entities = torch.cat(
            [
                flat_tokens[:, 1 : 1 + MAX_TEAM_SLOTS],
                self.no_target_embedding[None, None].expand(
                    time_steps * batch_size, 1, -1
                ),
            ],
            dim=1,
        )
        target_query = self.target_query(
            torch.cat(
                [flat_context, type_embedding, selected_source], dim=-1
            )
        )
        target_logits = torch.einsum(
            "bd,btd->bt", target_query, self.target_key(target_entities)
        ) / math.sqrt(self.config.d_model)
        target_logits = self._masked_logits(
            target_logits,
            self._target_mask(
                flat_observation["target_mask"], action_type, source
            ),
        )
        target = flat_actions[:, 2].long()
        target_dist = Categorical(logits=target_logits)

        log_prob = (
            type_dist.log_prob(action_type)
            + source_dist.log_prob(source)
            + target_dist.log_prob(target)
        ).reshape(time_steps, batch_size)
        entropy = (
            type_dist.entropy() + source_dist.entropy() + target_dist.entropy()
        ).reshape(time_steps, batch_size)
        if self.config.entropy_mode == "exact_joint":
            entropy = self.exact_primitive_entropy(
                flat_tokens, flat_context, flat_observation
            ).reshape(time_steps, batch_size)
        if self.end_freeze_head is not None:
            freeze, conditional_entropy = self._end_freeze_decision(
                flat_context, flat_tokens, flat_observation, type_dist, action_type,
                masks=end_freeze_masks.reshape(-1),
            )
            log_prob = log_prob + freeze.log_prob.reshape(time_steps, batch_size)
            entropy = entropy + conditional_entropy.reshape(time_steps, batch_size)
        value = self.value_head(flat_context).squeeze(-1).reshape(
            time_steps, batch_size
        )
        battle_logits = self.battle_head(flat_context).reshape(
            time_steps, batch_size, 3
        )
        board_strength = self.board_strength_head(flat_context).squeeze(-1).reshape(
            time_steps, batch_size
        )
        opponent_prediction = {
            key: value.reshape(1 if forecast_first_only else time_steps, batch_size, *value.shape[1:])
            for key, value in flat_opponent_prediction.items()
        }
        return SequencePolicyOutput(
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            memory_states=memory_states,
            final_memory=memory,
            battle_logits=battle_logits,
            board_strength=board_strength,
            **opponent_prediction,
        )

    def _source_entities(self, tokens, action_type):
        batch_size = tokens.shape[0]
        team = tokens[:, 1 : 1 + MAX_TEAM_SLOTS]
        team = torch.cat(
            [
                team,
                torch.zeros(
                    batch_size,
                    MAX_SHOP_SLOTS - MAX_TEAM_SLOTS,
                    self.config.d_model,
                    device=tokens.device,
                    dtype=tokens.dtype,
                ),
            ],
            dim=1,
        )
        shop_start = 1 + MAX_TEAM_SLOTS
        shop = tokens[:, shop_start : shop_start + MAX_SHOP_SLOTS]
        shop_type_lookup = torch.zeros(
            NUM_ACTION_TYPES, dtype=torch.bool, device=tokens.device
        )
        shop_type_lookup[list(self.SHOP_SOURCE_TYPES)] = True
        use_shop = shop_type_lookup[action_type][:, None, None]
        sources = torch.where(use_shop, shop, team)
        no_source = self.no_source_embedding[None, None].expand(
            batch_size, 1, -1
        )
        return torch.cat([sources, no_source], dim=1)

    @staticmethod
    def _conditional_mask(mask, first_index):
        batch = torch.arange(mask.shape[0], device=mask.device)
        return mask[batch, first_index].bool()

    @staticmethod
    def _target_mask(mask, action_type, source):
        batch = torch.arange(mask.shape[0], device=mask.device)
        return mask[batch, action_type, source].bool()

    @staticmethod
    def _masked_logits(logits, mask):
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("Every autoregressive stage needs one legal choice")
        return logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)

    @staticmethod
    def _select(distribution, actions, column, deterministic):
        if actions is not None:
            return actions[:, column].long()
        if deterministic:
            return distribution.logits.argmax(dim=-1)
        return distribution.sample()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
