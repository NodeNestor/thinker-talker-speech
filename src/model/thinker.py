"""Thinker — Qwen 3.5 0.8B wrapper with LoRA and hidden state extraction.

The Thinker operates entirely in text space. It receives speech features
from the Whisper adapter and generates text tokens. Hidden states from both
DeltaNet (linear attention) and standard attention layers are exposed for
the emotion probe and Talker connector.

Qwen 3.5 hybrid architecture per block:
  - 3x Gated DeltaNet layers (linear attention, recurrent state)
  - 1x Gated Attention layer (softmax, standard)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


QWEN_MODEL_ID = "Qwen/Qwen3.5-0.6B"  # Will update when 0.8B is available
# Fallback for initial dev (standard attention, proven compatible):
QWEN_FALLBACK_ID = "Qwen/Qwen3-0.6B"


class Thinker(nn.Module):
    """Qwen 3.5 Thinker with hidden state hooks for emotion probing."""

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_id = model_id

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # Need eager for hidden state hooks
        ).to(device)

        # Apply LoRA
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Hidden state storage (populated by hooks)
        self._hidden_states = {}
        self._deltanet_states = {}
        self._attention_states = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all transformer layers to capture hidden states.

        Qwen 3.5 alternates: DeltaNet, DeltaNet, DeltaNet, Attention (repeat).
        We capture both types separately for the emotion probe.
        """
        self._hooks = []

        for i, layer in enumerate(self.model.base_model.model.model.layers
                                  if hasattr(self.model, 'base_model')
                                  else self.model.model.layers):
            layer_type = "attention" if (i + 1) % 4 == 0 else "deltanet"

            def make_hook(layer_idx, ltype):
                def hook_fn(module, input, output):
                    # output is typically (hidden_states, ...) or a tuple
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    self._hidden_states[layer_idx] = hs.detach()
                    if ltype == "deltanet":
                        self._deltanet_states[layer_idx] = hs.detach()
                    else:
                        self._attention_states[layer_idx] = hs.detach()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(i, layer_type))
            self._hooks.append(handle)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        """Forward pass. Returns model output + captured hidden states."""
        # Clear previous states
        self._hidden_states.clear()
        self._deltanet_states.clear()
        self._attention_states.clear()

        output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        return output

    def get_deltanet_states(self) -> dict[int, torch.Tensor]:
        """Get DeltaNet layer hidden states (slow-moving context / mood)."""
        return dict(self._deltanet_states)

    def get_attention_states(self) -> dict[int, torch.Tensor]:
        """Get standard attention layer hidden states (per-token focus)."""
        return dict(self._attention_states)

    def get_last_hidden_state(self) -> torch.Tensor:
        """Get the final layer's hidden state for the Talker connector."""
        if self._hidden_states:
            max_layer = max(self._hidden_states.keys())
            return self._hidden_states[max_layer]
        return None

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def generate(self, **kwargs):
        """Text generation (for tool calls and text output)."""
        return self.model.generate(**kwargs)
