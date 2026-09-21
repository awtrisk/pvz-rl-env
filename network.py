"""
PvZActorCritic: PyTorch network for the Headless PvZ RL agent (Phase 4).

Observation (Dict):
    spatial: float32 (B, Seq, 5, 9, 36)   -- 3 plant channels + 33 zombie-type HP channels
    global:  float32 (B, Seq, 12)         -- sun / 9990, wave / total, 10 seed cooldowns

Architecture:
    1. Spatial Encoder: 3 CNN layers -> flatten -> Linear -> ReLU -> 256-dim
    2. Global Encoder:  2-layer MLP -> 64-dim
    3. Concatenation -> 320-dim feature vector per timestep
    4. Memory Layer:    mambapy.Mamba (d_model=320, n_layers=2) -> 320-dim
    5. Latent projection: Linear -> 512-dim (matches plan's "512-dim Latent Vector")
    6. Dual Heads:
        - Critic: Linear -> 1 scalar value estimate
        - Actor:  Linear -> 496 logits, then add -inf for illegal actions via mask

The mambapy.Mamba module provides both forward() for sequence training and
step() for cached single-step inference.  We expose both here.
"""

from itertools import pairwise
from math import prod

import torch
from mambapy.mamba import Mamba, MambaConfig
from torch import nn
from torch.distributions import Categorical


class FactorizedActor(nn.Module):
    """
    Factorized actor head for the 496-action PvZ space.

    The flat action space is encoded as:
        - 0: wait
        - 1..45: shovel at (row, col)  [row=0..4, col=0..8]
        - 46..495: plant seed s at (row, col)  [s=0..9, row=0..4, col=0..8]

    The actor is factorized into four independent heads:
        - mode:  {wait, shovel, plant}  (3)
        - seed:  which seed packet       (10, used only for plant)
        - row:   lane                     (5)
        - col:   column                   (9)

    The logit of a flat action is the sum of the relevant factor logits. This
    is much more parameter-efficient than a single (latent_dim, 496) matrix
    and allows the policy to generalize across seed/row/col choices.

    The entropy bonus is computed as the conditional entropy of the implied
    factorized distribution, so entropy does not spuriously count seed/row/col
    diversity when the sampled mode is wait. This makes entropy collapse harder.
    """

    NUM_MODES = 3
    NUM_SEEDS = 10
    NUM_ROWS = 5
    NUM_COLS = 9
    NUM_ACTIONS = 496

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.mode_head = nn.Linear(latent_dim, self.NUM_MODES)
        self.seed_head = nn.Linear(latent_dim, self.NUM_SEEDS)
        self.row_head = nn.Linear(latent_dim, self.NUM_ROWS)
        self.col_head = nn.Linear(latent_dim, self.NUM_COLS)
        # Migration seam: starts as an exact no-op, then learns arbitrary
        # seed-row-column interactions that the additive heads cannot express.
        self.residual = nn.Linear(latent_dim, self.NUM_ACTIONS)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

        # Precompute mapping from flat action index to (mode, seed, row, col).
        # seed is ignored for wait/shovel; row/col are ignored for wait.
        factors = torch.zeros(self.NUM_ACTIONS, 4, dtype=torch.long)
        factors[0, 0] = 0  # wait mode
        for row in range(self.NUM_ROWS):
            for col in range(self.NUM_COLS):
                idx = 1 + row * self.NUM_COLS + col  # shovel
                factors[idx, 0] = 1  # shovel mode
                factors[idx, 2] = row
                factors[idx, 3] = col
        for seed in range(self.NUM_SEEDS):
            for row in range(self.NUM_ROWS):
                for col in range(self.NUM_COLS):
                    idx = (
                        46
                        + seed * (self.NUM_ROWS * self.NUM_COLS)
                        + row * self.NUM_COLS
                        + col
                    )
                    factors[idx, 0] = 2  # plant mode
                    factors[idx, 1] = seed
                    factors[idx, 2] = row
                    factors[idx, 3] = col
        self.action_factors: torch.Tensor
        self.register_buffer("action_factors", factors)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (..., latent_dim)
        Returns flat logits: (..., NUM_ACTIONS)
        """
        mode_logits = self.mode_head(latent)  # (..., 3)
        seed_logits = self.seed_head(latent)  # (..., 10)
        row_logits = self.row_head(latent)  # (..., 5)
        col_logits = self.col_head(latent)  # (..., 9)

        # Gather relevant factor logit for each flat action.
        # Flat action a: logit[a] = mode_logit[mode_a]
        #               + (mode_a == plant ? seed_logit[seed_a] : 0)
        #               + (mode_a != wait ? row_logit[row_a] + col_logit[col_a] : 0)
        mode_a = self.action_factors[:, 0]  # (496,)
        seed_a = self.action_factors[:, 1]
        row_a = self.action_factors[:, 2]
        col_a = self.action_factors[:, 3]

        # Use advanced indexing with broadcast.
        # latent shape: (..., D). We want to gather for each action.
        # Reshape to (B, 1, D) for broadcasting, then index.
        leading_dims = latent.shape[:-1]
        flat_batch = prod(leading_dims)

        mode_contrib = mode_logits.reshape(flat_batch, self.NUM_MODES)[
            :, mode_a
        ]  # (B, 496)
        seed_contrib = seed_logits.reshape(flat_batch, self.NUM_SEEDS)[
            :, seed_a
        ]  # (B, 496)
        row_contrib = row_logits.reshape(flat_batch, self.NUM_ROWS)[
            :, row_a
        ]  # (B, 496)
        col_contrib = col_logits.reshape(flat_batch, self.NUM_COLS)[
            :, col_a
        ]  # (B, 496)

        plant_mask = (mode_a == 2).float()  # (496,)
        non_wait_mask = (mode_a != 0).float()

        logits = (
            mode_contrib
            + plant_mask * seed_contrib
            + non_wait_mask * (row_contrib + col_contrib)
        )
        logits = logits.reshape(*leading_dims, self.NUM_ACTIONS)
        return logits + self.residual(latent)

    def entropy(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Conditional factorized entropy of the implied factorized distribution.
        This is the standard entropy bonus for a multi-discrete / factorized
        action space: it sums the entropy of each independent factor, weighted
        by the probability that the factor is active. This prevents the mode
        head from collapsing to wait with near-zero entropy cost, because even
        a tiny amount of probability mass on non-wait modes forces row/col
        entropy, and the mode entropy itself is directly rewarded.

        Returns: (...,)
        """
        mode_logits = self.mode_head(latent)
        seed_logits = self.seed_head(latent)
        row_logits = self.row_head(latent)
        col_logits = self.col_head(latent)

        mode_dist = Categorical(logits=mode_logits)
        seed_dist = Categorical(logits=seed_logits)
        row_dist = Categorical(logits=row_logits)
        col_dist = Categorical(logits=col_logits)

        # P(plant) = mode probability at index 2
        p_plant = mode_dist.probs[..., 2]
        # P(wait) = mode probability at index 0
        p_wait = mode_dist.probs[..., 0]
        p_non_wait = 1.0 - p_wait

        return (
            3.0 * mode_dist.entropy()
            + p_plant * seed_dist.entropy()
            + p_non_wait * (row_dist.entropy() + col_dist.entropy())
        )


SPATIAL_SHAPE = (5, 9, 36)  # 3 plant channels + all 33 zombie types
LEGACY_GLOBAL_SIZE = 12
GLOBAL_SIZE = 24
NUM_ACTIONS = 496
CNN_HIDDEN_DIM = 256
GLOBAL_HIDDEN_DIM = 64
FEATURE_DIM = CNN_HIDDEN_DIM + GLOBAL_HIDDEN_DIM  # 320
LATENT_DIM = 512
MAMBA_LAYERS = 2


class PvZActorCritic(nn.Module):
    """
    Actor-critic network for PvZ: Survival Endless.

    Supports two calling patterns:
        - forward_sequence(spatial, global_vec, mask) -> action/logprob/entropy/value
          for full-trajectory PPO update batches (B, L, ...).
        - forward_step(spatial, global_vec, mask, caches) -> action/logprob/entropy/value, caches
          for single-step environment rollouts with recurrent state.
    """

    def __init__(
        self,
        spatial_shape: tuple[int, int, int] = SPATIAL_SHAPE,
        global_size: int = GLOBAL_SIZE,
        num_actions: int = NUM_ACTIONS,
        cnn_hidden_dim: int = CNN_HIDDEN_DIM,
        global_hidden_dim: int = GLOBAL_HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        mamba_layers: int = MAMBA_LAYERS,
    ) -> None:
        super().__init__()
        self.spatial_shape = spatial_shape
        self.global_size = global_size
        self.num_actions = num_actions
        self.feature_dim = cnn_hidden_dim + global_hidden_dim
        self.latent_dim = latent_dim

        rows, cols, in_channels = spatial_shape

        # 1. Spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * rows * cols, cnn_hidden_dim),
            nn.ReLU(),
        )

        # 2. Global encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(LEGACY_GLOBAL_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, global_hidden_dim),
            nn.ReLU(),
        )
        # New Markov-state fields enter through a zero adapter, preserving every
        # legacy checkpoint action until adaptation training changes it.
        self.state_adapter = nn.Linear(
            global_size - LEGACY_GLOBAL_SIZE, global_hidden_dim, bias=False
        )
        nn.init.zeros_(self.state_adapter.weight)

        # 4. Memory layer (mambapy pure-PyTorch Mamba)
        self.mamba_config = MambaConfig(d_model=self.feature_dim, n_layers=mamba_layers)
        self.mamba = Mamba(self.mamba_config)
        # Zero keeps legacy feed-forward checkpoints bit-identical. Training can
        # then admit only the temporal channels that improve the incumbent.
        self.memory_gate = nn.Parameter(torch.zeros(self.feature_dim))

        # 5. Latent projection to plan's 512-dim latent vector
        self.latent_proj = nn.Sequential(
            nn.Linear(self.feature_dim, latent_dim),
            nn.ReLU(),
        )

        # 6. Dual heads
        self.critic = nn.Linear(latent_dim, 1)
        self.actor = FactorizedActor(latent_dim)

    # ------------------------------------------------------------------
    # Shared feature extraction
    # ------------------------------------------------------------------
    def _encode_features(
        self, spatial: torch.Tensor, global_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a single observation (or flattened batch) into a feature vector.
        spatial:  (N, 5, 9, 36)
        global_vec: (N, 12)
        returns: (N, feature_dim)
        """
        N = spatial.shape[0]
        s = spatial.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
        s_feat = self.spatial_encoder(s)  # (N, 256)

        g = global_vec.reshape(N, self.global_size)
        g_feat = self.global_encoder(g[:, :LEGACY_GLOBAL_SIZE])
        g_feat = g_feat + self.state_adapter(g[:, LEGACY_GLOBAL_SIZE:])  # (N, 64)

        features = torch.cat([s_feat, g_feat], dim=-1)  # (N, 320)
        return features

    # ------------------------------------------------------------------
    # Sequence mode (for training / PPO updates)
    # ------------------------------------------------------------------
    def forward_sequence(
        self,
        spatial: torch.Tensor,
        global_vec: torch.Tensor,
        episode_starts: torch.Tensor | None = None,
        initial_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        Run the full feature + memory pipeline on a sequence batch.
        spatial:   (B, L, 5, 9, 36)
        global_vec:(B, L, 12)
        returns:   (B, L, latent_dim)
        """
        B, L = spatial.shape[:2]
        flat_spatial = spatial.reshape(B * L, *self.spatial_shape)
        flat_global = global_vec.reshape(B * L, self.global_size)

        features = self._encode_features(flat_spatial, flat_global)  # (B*L, 320)
        features = features.reshape(B, L, self.feature_dim)  # (B, L, 320)

        starts = (
            torch.zeros((B, L), dtype=torch.bool, device=features.device)
            if episode_starts is None
            else episode_starts.to(dtype=torch.bool, device=features.device)
        )
        if starts.shape != (B, L):
            raise ValueError(f"episode_starts must have shape {(B, L)}")

        if initial_caches is not None:
            caches = [(h.clone(), inputs.clone()) for h, inputs in initial_caches]
            outputs = []
            for index in range(L):
                keep = (~starts[:, index]).to(features.dtype).view(B, 1, 1)
                caches = [(h * keep, inputs * keep) for h, inputs in caches]
                recurrent_step, caches = self.mamba.step(features[:, index], caches)
                outputs.append(recurrent_step)
            recurrent = torch.stack(outputs, dim=1)
        elif episode_starts is None:
            recurrent = self.mamba(features)
        else:
            outputs = []
            for batch in range(B):
                boundaries = (
                    torch.nonzero(starts[batch], as_tuple=False).flatten().tolist()
                )
                if not boundaries or boundaries[0] != 0:
                    boundaries.insert(0, 0)
                boundaries.append(L)
                outputs.append(
                    torch.cat(
                        [
                            self.mamba(features[batch : batch + 1, begin:end])
                            for begin, end in pairwise(boundaries)
                            if begin < end
                        ],
                        dim=1,
                    )
                )
            recurrent = torch.cat(outputs, dim=0)

        # Mamba blocks already contain residual paths, so gate only their delta.
        # gate=0 exactly reproduces the feed-forward incumbent.
        memory_out = features + torch.tanh(self.memory_gate) * (recurrent - features)
        latent = self.latent_proj(memory_out)  # (B, L, 512)
        return latent

    def get_value(
        self, spatial: torch.Tensor, global_vec: torch.Tensor
    ) -> torch.Tensor:
        """Returns value estimates for a sequence batch. (B, L, 1)"""
        latent = self.forward_sequence(spatial, global_vec)
        return self.critic(latent)

    def get_action_and_value(
        self,
        spatial: torch.Tensor,
        global_vec: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        episode_starts: torch.Tensor | None = None,
        initial_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action (or evaluate a given action) and return value estimate.
        spatial:     (B, L, 5, 9, 36)
        global_vec:  (B, L, 12)
        action:      optional (B, L) long tensor to evaluate; takes precedence over deterministic.
        deterministic: select the masked argmax when no action is supplied.

        Returns: action, log_prob, entropy, value
        """
        latent = self.forward_sequence(
            spatial, global_vec, episode_starts, initial_caches
        )  # (B, L, 512)
        logits = self.actor(latent)  # (B, L, 496)
        logits = self._apply_action_mask(logits, action_mask)

        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()

        # Entropy must match the *masked flat distribution* used for sampling
        # and PPO log-probabilities. Factor-head entropy includes illegal
        # actions, so it optimizes a different policy whenever a seed cooldown
        # or curriculum restriction is active.
        entropy = dist.entropy()
        return action, dist.log_prob(action), entropy, self.critic(latent)

    # ------------------------------------------------------------------
    # Single-step mode (for environment rollouts with Mamba cache)
    # ------------------------------------------------------------------
    def init_caches(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Initialize Mamba caches for single-step rollout.
        Returns a list of (h, inputs) tuples, one per Mamba layer.
        h:      (B, ED, N)
        inputs: (B, ED, d_conv-1)
        """
        caches = []
        for _ in range(self.mamba_config.n_layers):
            h = torch.zeros(
                batch_size,
                self.mamba_config.d_inner,
                self.mamba_config.d_state,
                device=device,
            )
            inputs = torch.zeros(
                batch_size,
                self.mamba_config.d_inner,
                self.mamba_config.d_conv - 1,
                device=device,
            )
            caches.append((h, inputs))
        return caches

    def step_logits(
        self,
        spatial: torch.Tensor,
        global_vec: torch.Tensor,
        action_mask: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Advance recurrent state once and return masked logits and value."""
        features = self._encode_features(spatial, global_vec)
        recurrent, new_caches = self.mamba.step(features, caches)
        memory_out = features + torch.tanh(self.memory_gate) * (recurrent - features)
        latent = self.latent_proj(memory_out)
        return (
            self._apply_action_mask(self.actor(latent), action_mask),
            self.critic(latent),
            new_caches,
        )

    def forward_step(
        self,
        spatial: torch.Tensor,
        global_vec: torch.Tensor,
        action_mask: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Single-step forward pass. Returns (action, log_prob, entropy, value, new_caches).
        spatial:     (B, 5, 9, 36)
        global_vec:  (B, 12)
        action_mask: (B, 496) bool
        deterministic: select the masked argmax when no action is supplied.

        The zero-initialized residual gate preserves legacy behavior while the
        cached Mamba state learns useful temporal corrections.
        """
        logits, value, new_caches = self.step_logits(
            spatial, global_vec, action_mask, caches
        )
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()

        # Keep the exploration objective aligned with the legal-action
        # distribution used to sample this environment step.
        entropy = dist.entropy()
        return action, dist.log_prob(action), entropy, value, new_caches

    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Migrate legacy observations and policy heads without changing logits."""
        state_dict = dict(state_dict)
        state_dict.setdefault("memory_gate", torch.zeros_like(self.memory_gate))
        state_dict.setdefault(
            "state_adapter.weight", torch.zeros_like(self.state_adapter.weight)
        )
        state_dict.setdefault(
            "actor.residual.weight", torch.zeros_like(self.actor.residual.weight)
        )
        state_dict.setdefault(
            "actor.residual.bias", torch.zeros_like(self.actor.residual.bias)
        )

        conv_key = "spatial_encoder.0.weight"
        old_conv = state_dict.get(conv_key)
        current_conv = self.state_dict()[conv_key]
        if old_conv is not None and old_conv.shape != current_conv.shape:
            if old_conv.shape[1] != 33 or current_conv.shape[1] != 36:
                raise RuntimeError(
                    f"cannot migrate spatial encoder from {tuple(old_conv.shape)} "
                    f"to {tuple(current_conv.shape)}"
                )
            padded = torch.zeros_like(current_conv)
            padded[:, : old_conv.shape[1]] = old_conv
            state_dict[conv_key] = padded
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _apply_action_mask(
        logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask out illegal actions by setting their logits to -inf.
        action_mask: True for legal actions. Can be bool, int8, float.
        """
        mask = action_mask.to(logits.device)
        if logits.shape != mask.shape or logits.shape[-1] != NUM_ACTIONS:
            raise ValueError(
                f"logits and action mask must have identical (..., {NUM_ACTIONS}) "
                f"shapes, got {tuple(logits.shape)} and {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("every policy row must contain at least one legal action")
        logits = logits.masked_fill(~mask, -torch.inf)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
