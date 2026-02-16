from torch import nn, optim
from agent.encoder import *
from agent.policy import *
from agent.policy.distributors import *
from agent import *
from agent.buffer import *

def build_sequential(
        layer_configs: list,
        input_dim: int
) -> tuple[nn.Module, float]:
    """Creates a sequential module and detects output dimension."""
    layers = []
    current_dim = input_dim
    for cfg in layer_configs:
        layer_type = getattr(nn, cfg['type'])
        params = {k: v for k, v in cfg.items() if k != 'type'}

        if layer_type == nn.Linear:
            layer = layer_type(in_features=current_dim, **params)
            current_dim = params['out_features']
        else:
            layer = layer_type(**params)
        layers.append(layer)
    return nn.Sequential(*layers), current_dim

def build_optimizer(
        config_dict: dict,
        agent_components: dict
) -> optim.Optimizer:
    # Extract optimizer configuration and define learning rates
    opt_cfg = config_dict['hyper_params']['optimizer']
    opt_cls = getattr(optim, opt_cfg['type'])
    lrs_cfg = opt_cfg.get('lrs', {})
    default_lr = lrs_cfg.get('default', 1e-3)

    # Build a mapping of all modules and their sub-modules to assign specific learning rates
    module_dict = {}
    for comp_name, comp in agent_components.items():
        if isinstance(comp, nn.Module):
            module_dict[comp_name] = comp
            for child_name, child_module in comp.named_modules():
                if child_name:
                    module_dict[child_name] = child_module

    # Iterate through specified learning rates and assign parameters to groups
    param_groups = []
    seen_params = set()

    for target_name, lr in lrs_cfg.items():
        if target_name != 'default' and target_name in module_dict:
            target_module = module_dict[target_name]
            params_to_add = []
            for p in target_module.parameters():
                if p not in seen_params:
                    params_to_add.append(p)
                    seen_params.add(p)
            if params_to_add:
                param_groups.append({'params': params_to_add, 'lr': lr})

    # Gather remaining unassigned parameters and apply the default learning rate
    remaining_params = []
    for comp in agent_components.values():
        if isinstance(comp, nn.Module):
            for p in comp.parameters():
                if p not in seen_params:
                    remaining_params.append(p)
                    seen_params.add(p)

    if remaining_params:
        param_groups.append({'params': remaining_params, 'lr': default_lr})

    # Initialize the optimizer with the constructed parameter groups
    base_params = {k: v for k, v in opt_cfg.items() if k not in ['type', 'lrs']}

    return opt_cls(
        param_groups,
        **base_params
    )

def get_criterion(name: str):
    """Retrieves a loss function by name."""
    if not name: return None
    if hasattr(torch.nn.functional, name):
        return getattr(torch.nn.functional, name)
    if hasattr(torch, name):
        return getattr(torch, name)
    cls = getattr(nn, name, None)
    return cls() if cls else None

def build_agent(
        config_dict: dict,
        device: str = 'cpu'
) -> BaseAgent:
    # --- Part 1: Components (Encoder, Policy, Buffer) ---
    state_dim = config_dict['env']['state_dim']
    components = {}

    for name, cfg in config_dict['agent']['components'].items():
        cls_type = cfg['type']
        cls = globals().get(cls_type)
        if 'layers' in cfg:
            input_dim = cfg.get('input_dim', state_dim)
            net, out_dim = build_sequential(cfg['layers'], input_dim)
            params = cfg.get('params', {})
            if 'feature_dim' not in params: params['feature_dim'] = out_dim
            components[name] = cls(network=net, **params)
        elif 'components' in cfg:
            sub_components = {}
            feat_dim = cfg.get('feature_dim')
            for sub_name, sub_cfg in cfg['components'].items():
                if isinstance(sub_cfg, list):
                    sub_net, _ = build_sequential(sub_cfg, feat_dim)
                    sub_components[sub_name] = sub_net
                elif isinstance(sub_cfg, dict) and 'type' in sub_cfg:
                    sub_cls = globals().get(sub_cfg['type'])
                    sub_components[sub_name] = sub_cls(**sub_cfg.get('params', {}))
            components[name] = cls(**sub_components, **cfg.get('params', {}))
        else:
            components[name] = cls(**cfg.get('params', {}))

    # --- Part 2: Optimizer ---
    trainable_modules = {k: v for k, v in components.items() if isinstance(v, nn.Module)}
    optimizer = build_optimizer(config_dict, trainable_modules)

    # --- Part 3: Hyper-parameters & Criterion ---
    hyper_params = config_dict.get('hyper_params', {}).copy()
    if 'optimizer' in hyper_params: del hyper_params['optimizer']

    if 'criterion' in hyper_params:
        criteria_cfg = hyper_params.pop('criterion')
        for key, loss_name in criteria_cfg.items():
            if loss_name:
                loss_cls = getattr(nn, loss_name)
                hyper_params[key] = loss_cls()

    # --- Part 4: Build Agent ---
    agent_cls = globals()[config_dict['agent']['type']]
    return agent_cls(
        **components,
        optimizer=optimizer,
        device=device,
        **hyper_params
    )
