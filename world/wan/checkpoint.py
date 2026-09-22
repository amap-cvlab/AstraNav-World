"""Checkpoint layout resolution and strict ActionFormer loading."""
import os
import torch
from safetensors import safe_open

def resolve_checkpoint_paths(model_path):
    root = os.path.abspath(os.path.expanduser(model_path))
    qwen_path = os.path.join(root, "qwenmodel")
    if not os.path.isdir(qwen_path):
        qwen_path = root
    # Accept either the checkpoint root or its qwenmodel subdirectory.
    roots = [root]
    if os.path.basename(root) == "qwenmodel":
        roots.insert(0, os.path.dirname(root))
    elif qwen_path != root:
        roots.append(qwen_path)
    candidates = [os.path.join(directory, name) for directory in roots
                  for name in ("transformer_module.safetensors", "transformer_modules.pth")]
    action_path = next((path for path in candidates if os.path.isfile(path)), None)
    if action_path is None:
        raise FileNotFoundError("ActionFormer checkpoint not found. Checked: " + ", ".join(candidates))
    return qwen_path, action_path


def load_action_former_checkpoint(action_former, checkpoint_path):
    if checkpoint_path.endswith(".safetensors"):
        # Action-only mode reads only the head tensors, not the Wan LoRA/projector tensors.
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            prefixed = any(key.startswith("action_former.") for key in keys)
            state = {key: checkpoint.get_tensor(key) for key in keys
                     if (key.startswith("action_former.") if prefixed else key in action_former.state_dict())}
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict in {checkpoint_path}")

    expected = action_former.state_dict()
    prefixed = any(key.startswith("action_former.") for key in state)
    supplied = {key.removeprefix("action_former."): value for key, value in state.items()
                if key.startswith("action_former.")} if prefixed else state
    missing = sorted(set(expected) - set(supplied))
    mismatched = [f"{key}: expected {tuple(value.shape)}, got "
                  f"{tuple(supplied[key].shape) if torch.is_tensor(supplied[key]) else type(supplied[key]).__name__}"
                  for key, value in expected.items() if key in supplied
                  and (not torch.is_tensor(supplied[key]) or supplied[key].shape != value.shape)]
    matched = {key: value for key, value in supplied.items() if key in expected
               and torch.is_tensor(value) and value.shape == expected[key].shape}
    total_numel = sum(value.numel() for value in expected.values())
    loaded_numel = sum(expected[key].numel() for key in matched)
    print(f"[ActionFormer] {checkpoint_path}: keys={len(matched)}/{len(expected)}, "
          f"elements={loaded_numel}/{total_numel}, "
          f"coverage={loaded_numel / max(total_numel, 1):.2%}, "
          f"ignored_checkpoint_keys={len(state) - len(matched)}")
    if missing or mismatched:
        raise RuntimeError(f"Incomplete ActionFormer checkpoint: {checkpoint_path}\n"
                           f"Missing keys: {missing}\nShape mismatches: {mismatched}")
    # Extra Wan/policy-head weights do not participate in this inference path.
    action_former.load_state_dict(matched, strict=True)
