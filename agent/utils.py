import torch
from torch import nn
from dataclasses import is_dataclass, asdict

def _get_optimizer_info(agent):
    opt = getattr(agent, "optimizer", None)
    if not opt:
        return "None"

    param_to_path = {}
    for m_name, module in agent.named_modules():
        if m_name == "":
            continue
        for p in module.parameters(recurse=False):
            param_to_path[id(p)] = m_name

    lines = []
    defaults = opt.defaults
    for i, group in enumerate(opt.param_groups):
        group_labels = {param_to_path.get(id(p)) for p in group['params']}
        group_labels.discard(None)
        
        label_str = f"\[{', '.join(sorted(group_labels))}]" if group_labels else f"Group {i}"

        hyper_params = [
            f"{k}={v}" for k, v in group.items()
            if k != 'params' and not k.startswith('_') and
            (k == 'lr' or (k in defaults and v != defaults[k]))
        ]
        lines.append(f"  ({', '.join(hyper_params)}):\n    {label_str}")
    return f"{opt.__class__.__name__}()\n" + "\n".join(lines)

def get_agent_summary(agent: nn.Module) -> str:
    """Generate a clean markup summary string for an agent."""
    lines = [f"[bold cyan][{agent.__class__.__name__} Summary][/bold cyan]"]
    lines.append(f"Device: \'{agent.device}\'")
    lines.append(f"Optimizer: {_get_optimizer_info(agent=agent)}")

    lines.append("\n[bold yellow][Architecture][/bold yellow]")
    lines.append(str(agent))

    if hasattr(agent, 'cfg') and agent.cfg:
        lines.append("\n[bold yellow][Configuration][/bold yellow]")
        cfg_dict = asdict(agent.cfg) if is_dataclass(agent.cfg) else getattr(agent.cfg, '__dict__', {})
        for k, v in sorted(cfg_dict.items()):
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)

class MetricTracker:
    def __init__(self, weights: dict = {}):
        self.metric_weights = weights
        self.reset()

    def reset(self):
        self._data = {}

    def store(self, **metrics):
        for k, v in metrics.items():
            if k not in self._data:
                self._data[k] = []
            val = v.detach().cpu().item() if torch.is_tensor(v) else v

            if k in self.metric_weights:
                val *= self.metric_weights[k]

            self._data[k].append(val)

    def result(self) -> dict:
        return {
            k: sum(v) / len(v) for k, v in self._data.items() if len(v) > 0
        }
