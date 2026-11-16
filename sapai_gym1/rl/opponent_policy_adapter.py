"""Opt-in causal residual experiment. Not installed in production training.

The old model/history/adapter and GRU are retained exactly. New context is
explicit, separately collated, and never appended to the old attention set.
"""
from dataclasses import asdict, replace

import torch
from torch import nn

ADAPTER_SCHEMA = 'causal-opponent-policy-residual-v1-20260906'


class CausalOpponentPolicy(nn.Module):
    def __init__(self, base_policy, forecast_model):
        super().__init__()
        self.base_policy = base_policy
        self.forecast_model = forecast_model
        self.projection = nn.Linear(forecast_model.config.width, base_policy.config.d_model)
        nn.init.zeros_(self.projection.weight); nn.init.zeros_(self.projection.bias)
        self.set_stage('forecast')

    @property
    def config(self):
        return self.base_policy.config

    def set_stage(self, stage):
        if stage not in ('forecast', 'policy_control', 'policy_residual'):
            raise ValueError('Unknown separately declared experiment stage')
        self.stage = stage
        self.base_policy.requires_grad_(stage != 'forecast')
        self.forecast_model.requires_grad_(stage == 'forecast')
        self.projection.requires_grad_(stage == 'policy_residual')

    def initial_memory(self, batch_size, device=None):
        return self.base_policy.initial_memory(batch_size, device)

    def _residual(self, causal_context):
        if causal_context is None:
            raise ValueError('Enabled residual requires explicit causal public context')
        # Stage 2 initially freezes the calibrated forecast representation. No
        # hidden forecast-head losses silently alter the policy-strength A/B.
        with torch.no_grad():
            encoded = self.forecast_model.encode_context(causal_context)
        return self.projection(encoded)

    def encode_decision(self, observation, memory=None, *, causal_context=None, forecast=True):
        features = self.base_policy.encode_decision(observation, memory, forecast=forecast)
        if self.stage == 'policy_residual':
            residual = self._residual(causal_context)
            if residual.shape != features.context.shape or not torch.isfinite(residual).all():
                raise ValueError('Invalid causal policy residual')
            features = replace(features, context=features.context+residual)
        return features

    def act(self, observation, memory=None, *, causal_context=None, deterministic=False):
        features = self.encode_decision(observation, memory, causal_context=causal_context, forecast=False)
        return self.base_policy.sample_joint_action(features, deterministic=deterministic)

    def forward(self, observation, memory=None, actions=None, deterministic=False, *, causal_context=None, **kwargs):
        features = self.encode_decision(observation, memory, causal_context=causal_context)
        return self.base_policy.decode_decision(features, actions, deterministic, **kwargs)

    def value_only(self, observation, memory=None, *, causal_context=None):
        features = self.encode_decision(observation, memory, causal_context=causal_context, forecast=False)
        return self.base_policy.value_from_encoded(features)

    def evaluate_sequence(self, observation, initial_memory, actions, *, causal_context=None, **kwargs):
        residual = None
        if self.stage == 'policy_residual':
            residual = self._residual(causal_context).reshape(*actions.shape[:2], self.config.d_model)
        return self.base_policy.evaluate_sequence(observation, initial_memory, actions, context_residual=residual, **kwargs)

    def forecast_loss(self, causal_context, next_public_board):
        if self.stage != 'forecast':
            raise ValueError('Forecast training must be a separate declared stage')
        return -self.forecast_model.log_prob(causal_context, next_public_board).mean()

    def export_metadata(self):
        return {'adapterSchema': ADAPTER_SCHEMA, 'stage': self.stage,
                'baseConfig': asdict(self.base_policy.config), 'forecastConfig': asdict(self.forecast_model.config),
                'activationRequiresIndependentQualityGate': True}
