from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from models.bi_privacy_defense import grad_reverse
from models.fl_split_bottleneck import BottleneckFedAvgView, NestedOrthogonalBottleneck


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation=nn.ReLU,
    output_activation=None,
) -> nn.Sequential:
    layers = []
    prev_dim = int(input_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(h)))
        layers.append(activation())
        prev_dim = int(h)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def init_linear_weights(module: nn.Module) -> nn.Module:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.0)
    return module


class LocalBIEncoder(nn.Module):
    """Local obs/action processor plus GRU-based BI extractor.

    Raw obs/action tensors are consumed only by this local module. The server
    receives only the returned BI sequence.
    """

    @staticmethod
    def build_bi_extractor_modules(
        local_pre_out_dim: int,
        bi_hidden_dim: int,
        gru_num_layers: int = 1,
        bi_extractor: str = "gru",
        bi_post_gru_mlp: bool = False,
        *,
        activate_feature_outputs: bool = False,
        init_seed: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Module]:
        """Create a BI extractor stack with the canonical initialization contract.

        The returned pair is ``(bi_extractor, f_B)``.  This helper is used for
        both independent and hard-shared extractor construction so the same
        module topology can be cloned or shared without reimplementing the
        initialization logic at call sites.
        """

        feature_output_activation = nn.ReLU if activate_feature_outputs else None
        if bi_extractor not in {"gru", "mlp", "none"}:
            raise ValueError(f"Unsupported bi_extractor={bi_extractor!r}; expected gru/mlp/none")
        if bi_post_gru_mlp and bi_extractor != "gru":
            raise ValueError("bi_post_gru_mlp requires bi_extractor='gru'")
        init_context = torch.random.fork_rng(devices=[], enabled=init_seed is not None)
        with init_context:
            if init_seed is not None:
                torch.manual_seed(int(init_seed))
            if bi_extractor == "gru":
                extractor = nn.GRU(
                    input_size=int(local_pre_out_dim),
                    hidden_size=int(bi_hidden_dim),
                    num_layers=int(gru_num_layers),
                    batch_first=True,
                )
                if bi_post_gru_mlp:
                    post = build_mlp(
                        input_dim=int(bi_hidden_dim),
                        hidden_dims=(int(bi_hidden_dim),),
                        output_dim=int(bi_hidden_dim),
                        output_activation=None,
                    )
                    init_linear_weights(post)
                else:
                    post = nn.Identity()
            elif bi_extractor == "mlp":
                extractor = build_mlp(
                    input_dim=int(local_pre_out_dim),
                    hidden_dims=(),
                    output_dim=int(bi_hidden_dim),
                    output_activation=feature_output_activation,
                )
                init_linear_weights(extractor)
                post = nn.Identity()
            else:
                extractor = nn.Identity()
                post = nn.Identity()
        return extractor, post

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        local_pre_hidden_dims: Sequence[int] = (128,),
        local_pre_out_dim: int = 128,
        bi_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        activate_feature_outputs: bool = False,
        bi_extractor: str = "gru",
        bi_post_gru_mlp: bool = False,
        action_pathway: str = "on",
        fuse: str = "learned",
        bottleneck_dim: Optional[int] = None,
        bottleneck_seed: int = 0,
        shared_bi_extractor: Optional[nn.Module] = None,
        shared_f_B: Optional[nn.Module] = None,
        bi_init_seed: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.local_pre_out_dim = int(local_pre_out_dim)
        self.bi_hidden_dim = int(bi_hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        self.bi_extractor_kind = str(bi_extractor)
        self.bi_post_gru_mlp = bool(bi_post_gru_mlp)
        self.action_pathway = str(action_pathway)
        self.fuse = str(fuse)
        self.bottleneck_dim = None if bottleneck_dim is None else int(bottleneck_dim)
        self.bottleneck_seed = int(bottleneck_seed)
        if self.bi_extractor_kind not in {"gru", "mlp", "none"}:
            raise ValueError(f"Unsupported bi_extractor={bi_extractor!r}; expected gru/mlp/none")
        if self.bi_post_gru_mlp and self.bi_extractor_kind != "gru":
            raise ValueError("bi_post_gru_mlp requires bi_extractor='gru'")
        if self.action_pathway not in {"on", "off"}:
            raise ValueError(f"Unsupported action_pathway={action_pathway!r}; expected on/off")
        if self.fuse not in {"learned", "concat"}:
            raise ValueError(f"Unsupported fuse={fuse!r}; expected learned/concat")
        if self.bottleneck_dim is not None:
            if self.bottleneck_dim <= 0:
                raise ValueError(f"bottleneck_dim must be positive, got {self.bottleneck_dim}")
            if self.bottleneck_dim > self.bi_hidden_dim:
                raise ValueError(
                    f"bottleneck_dim must be <= bi_hidden_dim, got bottleneck_dim={self.bottleneck_dim} "
                    f"bi_hidden_dim={self.bi_hidden_dim}"
                )
        feature_output_activation = nn.ReLU if activate_feature_outputs else None
        processor_input_dim = self.obs_dim + (self.act_dim if self.action_pathway == "on" else 0)

        if self.fuse == "learned":
            self.obs_action_processor = build_mlp(
                input_dim=processor_input_dim,
                hidden_dims=local_pre_hidden_dims,
                output_dim=self.local_pre_out_dim,
                output_activation=feature_output_activation,
            )
            init_linear_weights(self.obs_action_processor)
        else:
            # Direct concat ablation: keep the GRU/MLP input width fixed by
            # zero-padding the raw obs(+action) vector to local_pre_out_dim.
            # This avoids changing cross-client FedAvg state_dict shapes.
            if processor_input_dim > self.local_pre_out_dim:
                raise ValueError(
                    "fuse='concat' requires processor input dim <= local_pre_out_dim "
                    f"for zero-padded direct concat, got {processor_input_dim} > {self.local_pre_out_dim}"
                )
            self.obs_action_processor = nn.Identity()

        if (shared_bi_extractor is None) != (shared_f_B is None):
            raise ValueError("shared_bi_extractor and shared_f_B must either both be provided or both be omitted")
        if shared_bi_extractor is not None:
            if self.bi_extractor_kind != "gru":
                raise ValueError("shared_bi_extractor is only supported when bi_extractor='gru'")
            if self.bi_post_gru_mlp and not isinstance(shared_f_B, nn.Module):
                raise ValueError("shared_f_B is required when bi_post_gru_mlp=True")
            self.bi_extractor = shared_bi_extractor
            self.f_B = shared_f_B if shared_f_B is not None else nn.Identity()
        else:
            self.bi_extractor, self.f_B = self.build_bi_extractor_modules(
                local_pre_out_dim=self.local_pre_out_dim,
                bi_hidden_dim=self.bi_hidden_dim,
                gru_num_layers=self.gru_num_layers,
                bi_extractor=self.bi_extractor_kind,
                bi_post_gru_mlp=self.bi_post_gru_mlp,
                activate_feature_outputs=activate_feature_outputs,
                init_seed=bi_init_seed,
            )
        self.bottleneck: Optional[NestedOrthogonalBottleneck]
        if self.bottleneck_dim is None:
            self.bottleneck = None
        else:
            self.bottleneck = NestedOrthogonalBottleneck(
                feature_dim=self.bi_hidden_dim,
                bottleneck_dim=self.bottleneck_dim,
                seed=self.bottleneck_seed,
            )

    @staticmethod
    def _to_sequence(x: torch.Tensor) -> Tuple[torch.Tensor, str]:
        if x.dim() == 1:
            return x.unsqueeze(0).unsqueeze(0), "single"
        if x.dim() == 2:
            return x.unsqueeze(1), "batch"
        if x.dim() == 3:
            return x, "sequence"
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    def _format_hidden(
        self,
        hidden: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if hidden is None:
            return torch.zeros(
                self.gru_num_layers,
                batch_size,
                self.bi_hidden_dim,
                device=device,
                dtype=dtype,
            )
        if hidden.dim() == 1:
            if hidden.shape[0] != self.bi_hidden_dim:
                raise ValueError(f"Hidden dim mismatch: expected {self.bi_hidden_dim}, got {hidden.shape[0]}")
            h = hidden.view(1, 1, self.bi_hidden_dim).to(device=device, dtype=dtype)
            if batch_size != 1:
                h = h.expand(1, batch_size, self.bi_hidden_dim).contiguous()
        elif hidden.dim() == 2:
            if tuple(hidden.shape) != (batch_size, self.bi_hidden_dim):
                raise ValueError(f"Hidden shape mismatch: expected ({batch_size}, {self.bi_hidden_dim}), got {tuple(hidden.shape)}")
            h = hidden.unsqueeze(0).to(device=device, dtype=dtype)
        elif hidden.dim() == 3:
            expected = (self.gru_num_layers, batch_size, self.bi_hidden_dim)
            if tuple(hidden.shape) != expected:
                raise ValueError(f"Hidden shape mismatch: expected {expected}, got {tuple(hidden.shape)}")
            h = hidden.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported hidden shape: {tuple(hidden.shape)}")
        if self.gru_num_layers > 1 and h.shape[0] == 1:
            h = h.expand(self.gru_num_layers, batch_size, self.bi_hidden_dim).contiguous()
        return h

    @staticmethod
    def _restore_hidden_shape(h: torch.Tensor, mode: str) -> torch.Tensor:
        h_last = h[-1]
        if mode == "single":
            return h_last.squeeze(0)
        if mode in ("batch", "sequence"):
            return h_last
        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _pad_or_trim_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = x.shape[-1]
        if current_dim == target_dim:
            return x
        if current_dim > target_dim:
            return x[..., :target_dim]
        pad_width = target_dim - current_dim
        pad = torch.zeros(*x.shape[:-1], pad_width, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    @staticmethod
    def _last_step_as_hidden(bi_seq: torch.Tensor, mode: str) -> torch.Tensor:
        h_last = bi_seq[:, -1, :]
        if mode == "single":
            return h_last.squeeze(0)
        if mode in ("batch", "sequence"):
            return h_last
        raise ValueError(f"Unknown mode: {mode}")

    def encode_bi(
        self,
        obs: torch.Tensor,
        prev_act: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        *,
        return_transmitted: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_seq, obs_mode = self._to_sequence(obs)
        act_seq, act_mode = self._to_sequence(prev_act)
        if obs_mode != act_mode:
            raise ValueError(f"obs and action modes differ: {obs_mode} vs {act_mode}")
        if obs_seq.shape[:2] != act_seq.shape[:2]:
            raise ValueError(f"obs/action batch or sequence mismatch: {tuple(obs_seq.shape)} vs {tuple(act_seq.shape)}")
        if obs_seq.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected obs dim {self.obs_dim}, got {obs_seq.shape[-1]}")
        if act_seq.shape[-1] != self.act_dim:
            raise ValueError(f"Expected action dim {self.act_dim}, got {act_seq.shape[-1]}")
        batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
        device, dtype = obs_seq.device, obs_seq.dtype
        if self.action_pathway == "on":
            x = torch.cat([obs_seq, act_seq], dim=-1)
        else:
            x = obs_seq
        x_flat = x.reshape(batch_size * seq_len, -1)
        if self.fuse == "learned":
            local_flat = self.obs_action_processor(x_flat)
        else:
            local_flat = self._pad_or_trim_last_dim(x_flat, self.local_pre_out_dim)
        local_seq = local_flat.reshape(batch_size, seq_len, self.local_pre_out_dim)
        if self.bi_extractor_kind == "gru":
            h0 = self._format_hidden(hidden, batch_size=batch_size, device=device, dtype=dtype)
            bi_seq, hn = self.bi_extractor(local_seq, h0)
            hidden_out = self._restore_hidden_shape(hn, obs_mode)
        if self.bi_extractor_kind == "mlp":
            bi_flat = self.bi_extractor(local_seq.reshape(batch_size * seq_len, self.local_pre_out_dim))
            bi_seq = bi_flat.reshape(batch_size, seq_len, self.bi_hidden_dim)
            hidden_out = self._last_step_as_hidden(bi_seq, obs_mode)
        elif self.bi_extractor_kind == "none":
            bi_seq = self._pad_or_trim_last_dim(local_seq, self.bi_hidden_dim)
            hidden_out = self._last_step_as_hidden(bi_seq, obs_mode)
        else:
            hidden_out = self._restore_hidden_shape(hn, obs_mode)

        bi_seq = self.f_B(bi_seq)

        if self.bottleneck is None:
            transmitted_seq = bi_seq
            decoded_seq = bi_seq
        else:
            decoded_seq, transmitted_seq = self.bottleneck(bi_seq)
        if return_transmitted:
            return decoded_seq, hidden_out, transmitted_seq
        return decoded_seq, hidden_out

    @torch.no_grad()
    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if batch_size == 1:
            return torch.zeros(self.bi_hidden_dim, device=device, dtype=dtype)
        return torch.zeros(batch_size, self.bi_hidden_dim, device=device, dtype=dtype)

    def fedavg_bi_module(self) -> nn.Module:
        if self.bottleneck is None:
            return self.bi_extractor
        return BottleneckFedAvgView(self.bi_extractor, self.bottleneck)

    def local_parameter_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [self.obs_action_processor]
        if self.bottleneck is not None:
            modules.append(self.bottleneck)
        return modules

    def extractor_parameter_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = []
        if not isinstance(self.bi_extractor, nn.Identity):
            modules.append(self.bi_extractor)
        if not isinstance(self.f_B, nn.Identity):
            modules.append(self.f_B)
        return modules

    def all_parameter_modules(self) -> List[nn.Module]:
        return self.local_parameter_modules() + self.extractor_parameter_modules()


class GlobalBIServerCritic(nn.Module):
    """Server-side BI aggregator and value module.

    ``forward_bi`` is intentionally strict: it accepts BI tensors only. The
    trainer owns returns, value losses, advantages, and gradient handoff.
    """

    def __init__(
        self,
        num_agents: int = 3,
        bi_dim: int = 128,
        gat_hidden_dim: int = 128,
        gat_heads: int = 1,
        readout_hidden_dims: Sequence[int] = (128,),
        readout_dim: int = 128,
        value_hidden_dims: Sequence[int] = (64,),
        use_self_loops: bool = True,
        gat_dropout: float = 0.0,
        activate_feature_outputs: bool = False,
        aggregation: str = "gat",
        bi_type_grl: bool = False,
        bi_type_loss_coef: float = 0.0,
        bi_grl_alpha: float = 1.0,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.bi_dim = int(bi_dim)
        self.gat_hidden_dim = int(gat_hidden_dim)
        self.gat_heads = int(gat_heads)
        self.readout_dim = int(readout_dim)
        self.use_self_loops = bool(use_self_loops)
        self.aggregation = str(aggregation)
        self.bi_type_grl = bool(bi_type_grl)
        self.bi_type_loss_coef = float(bi_type_loss_coef)
        self.bi_grl_alpha = float(bi_grl_alpha)
        if self.bi_type_grl and self.bi_type_loss_coef <= 0.0:
            raise ValueError("bi_type_grl requires bi_type_loss_coef > 0.0")
        if self.aggregation not in {"gat", "mean", "none"}:
            raise ValueError(f"Unsupported aggregation={aggregation!r}; expected gat/mean/none")
        if self.aggregation != "gat" and self.bi_dim != self.gat_hidden_dim:
            raise ValueError(
                "aggregation mean/none requires bi_dim == gat_hidden_dim so the value path "
                f"keeps a stable width, got {self.bi_dim} vs {self.gat_hidden_dim}"
            )
        feature_output_activation = nn.ReLU if activate_feature_outputs else None

        if self.aggregation == "gat":
            self.gat = GATConv(
                in_channels=self.bi_dim,
                out_channels=self.gat_hidden_dim,
                heads=self.gat_heads,
                concat=False,
                dropout=gat_dropout,
                add_self_loops=self.use_self_loops,
            )
            self.gat_activation = nn.ReLU()
        else:
            self.gat = None
            self.gat_activation = None
        if self.bi_type_grl or self.bi_type_loss_coef != 0.0:
            self.type_predictor = build_mlp(
                input_dim=self.bi_dim,
                hidden_dims=(self.bi_dim,),
                output_dim=self.num_agents,
            )
            init_linear_weights(self.type_predictor)
        else:
            self.type_predictor = None
        if self.aggregation != "none":
            self.readout_net = build_mlp(
                input_dim=self.num_agents * self.gat_hidden_dim,
                hidden_dims=readout_hidden_dims,
                output_dim=self.readout_dim,
                output_activation=feature_output_activation,
            )
            init_linear_weights(self.readout_net)
        else:
            self.readout_net = None
        value_input_dim = self.gat_hidden_dim + self.readout_dim
        if len(value_hidden_dims) > 0:
            self.value_body = build_mlp(
                input_dim=value_input_dim,
                hidden_dims=value_hidden_dims[:-1],
                output_dim=value_hidden_dims[-1],
                output_activation=feature_output_activation,
            )
            init_linear_weights(self.value_body)
            body_out_dim = int(value_hidden_dims[-1])
        else:
            self.value_body = nn.Identity()
            body_out_dim = value_input_dim
        self.value_head = nn.Linear(body_out_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def _build_edge_index(self, device: torch.device) -> torch.Tensor:
        edges = []
        for src in range(self.num_agents):
            for dst in range(self.num_agents):
                if src != dst:
                    edges.append([src, dst])
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def _validate_bi_list(self, bi_list: List[torch.Tensor]) -> Tuple[int, int, torch.device, torch.dtype]:
        if not isinstance(bi_list, list):
            raise TypeError("forward_bi expects a list of BI tensors")
        if len(bi_list) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} BI tensors, got {len(bi_list)}")
        ref = bi_list[0]
        if not torch.is_tensor(ref):
            raise TypeError("BI entries must be tensors")
        if ref.dim() != 3:
            raise ValueError(f"BI tensors must have shape (B, T, H), got {tuple(ref.shape)}")
        batch_size, seq_len, hidden = ref.shape
        if hidden != self.bi_dim:
            raise ValueError(f"Expected BI hidden dim {self.bi_dim}, got {hidden}")
        for idx, bi in enumerate(bi_list):
            if not torch.is_tensor(bi):
                raise TypeError(f"BI entry {idx} is not a tensor")
            if tuple(bi.shape) != (batch_size, seq_len, self.bi_dim):
                raise ValueError(f"BI entry {idx} shape mismatch: expected {(batch_size, seq_len, self.bi_dim)}, got {tuple(bi.shape)}")
            if bi.dtype != ref.dtype:
                raise ValueError(f"BI entry {idx} dtype mismatch: {ref.dtype} vs {bi.dtype}")
            if bi.device != ref.device:
                raise ValueError(f"BI entry {idx} device mismatch: {ref.device} vs {bi.device}")
        return batch_size, seq_len, ref.device, ref.dtype

    def forward_bi(self, bi_list: List[torch.Tensor]):
        batch_size, seq_len, device, _ = self._validate_bi_list(bi_list)
        if self.aggregation == "gat":
            if self.gat is None or self.gat_activation is None:
                raise RuntimeError("aggregation='gat' requires registered GAT modules")
            edge_index = self._build_edge_index(device)
            node_rows = []
            for b in range(batch_size):
                time_rows = []
                for t in range(seq_len):
                    x_bt = torch.stack([bi_list[i][b, t] for i in range(self.num_agents)], dim=0)
                    gat_out = self.gat_activation(self.gat(x_bt, edge_index))
                    time_rows.append(gat_out)
                node_rows.append(torch.stack(time_rows, dim=0))
            node_embeddings_all = torch.stack(node_rows, dim=0)  # (B, T, N, H_gat)
            node_embeddings = [node_embeddings_all[:, :, i, :] for i in range(self.num_agents)]
            joint_nodes = torch.cat(node_embeddings, dim=-1)
            readout_flat = self.readout_net(joint_nodes.reshape(batch_size * seq_len, -1))
            readout = readout_flat.reshape(batch_size, seq_len, self.readout_dim)
        elif self.aggregation == "mean":
            mean_bi = torch.stack(bi_list, dim=0).mean(dim=0)
            node_embeddings = [mean_bi for _ in range(self.num_agents)]
            joint_nodes = torch.cat(node_embeddings, dim=-1)
            readout_flat = self.readout_net(joint_nodes.reshape(batch_size * seq_len, -1))
            readout = readout_flat.reshape(batch_size, seq_len, self.readout_dim)
        else:
            node_embeddings = list(bi_list)
            joint_nodes = torch.cat(node_embeddings, dim=-1)
            readout = torch.zeros(
                batch_size,
                seq_len,
                self.readout_dim,
                device=device,
                dtype=bi_list[0].dtype,
            )
        values = []
        for i in range(self.num_agents):
            value_input = torch.cat([node_embeddings[i], readout], dim=-1)
            body = self.value_body(value_input.reshape(batch_size * seq_len, -1))
            v = self.value_head(body).squeeze(-1).reshape(batch_size, seq_len)
            values.append(v)
        aux = {
            "node_embeddings": node_embeddings,
            "joint_nodes": joint_nodes,
            "readout": readout,
            "aggregation": self.aggregation,
        }
        return values, aux

    def _build_type_mixed_batch(self, bi_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[int], int, int]:
        batch_size, seq_len, device, _ = self._validate_bi_list(bi_list)
        flat_per_agent = batch_size * seq_len
        mixed = torch.cat([bi.reshape(flat_per_agent, self.bi_dim) for bi in bi_list], dim=0)
        labels = torch.cat(
            [
                torch.full((flat_per_agent,), i, dtype=torch.long, device=device)
                for i in range(self.num_agents)
            ],
            dim=0,
        )
        counts = [flat_per_agent for _ in range(self.num_agents)]
        return mixed, labels, counts, batch_size, seq_len

    def forward_type_branch(
        self,
        bi_list: List[torch.Tensor],
        *,
        use_grl: Optional[bool] = None,
        alpha: Optional[float] = None,
    ) -> dict:
        if self.type_predictor is None:
            raise RuntimeError(
                "Type predictor is disabled. Enable bi_type_grl or bi_type_loss_coef at construction time "
                "before using forward_type_branch."
            )
        mixed_bi, labels, counts, batch_size, seq_len = self._build_type_mixed_batch(bi_list)
        use_grl = self.bi_type_grl if use_grl is None else bool(use_grl)
        alpha = self.bi_grl_alpha if alpha is None else float(alpha)
        predictor_input = grad_reverse(mixed_bi, alpha) if use_grl else mixed_bi
        logits = self.type_predictor(predictor_input)
        loss = F.cross_entropy(logits, labels)
        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
            "mixed_bi": mixed_bi,
            "class_counts": counts,
            "samples_per_class": counts,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "use_grl": bool(use_grl),
            "alpha": float(alpha),
            "num_classes": self.num_agents,
        }
