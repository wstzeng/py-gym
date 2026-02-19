import torch
from torch import nn, optim
import agent
import agent.encoder
import agent.policy
import agent.policy.action_handler
import agent.buffer
from utils.logger import logger

def get_component_registry() -> dict:
    # Build registry from sub-modules' __all__
    registry = {
        "Sequential": nn.Sequential,
    }

    # Modules to scan for exported classes
    modules = [
        agent,
        agent.encoder,
        agent.policy,
        agent.policy.action_handler,
        agent.buffer
    ]

    for mod in modules:
        if hasattr(mod, "__all__"):
            for name in mod.__all__:
                registry[name] = getattr(mod, name)
        else:
            logger.warning(f"Module {mod.__name__} has no __all__ defined.")

    return registry

# Initialize the internal registry
COMPONENT_REGISTRY = get_component_registry()

def build_sequential(
        layer_configs: list,
        input_dim: int
) -> tuple[nn.Module, float]:
    """Creates a sequential module and detects output dimension."""
    layers = []
    current_dim = input_dim
    for cfg in layer_configs:
        # Get layer from torch.nn namespace
        layer_type = getattr(nn, cfg['type'])
        params = {k: v for k, v in cfg.items() if k != 'type'}

        if layer_type == nn.Linear:
            layer = layer_type(
                in_features=current_dim,
                **params
            )
            current_dim = params['out_features']
        else:
            layer = layer_type(**params)
        layers.append(layer)
    return nn.Sequential(*layers), current_dim

def build_optimizer(
        config_dict: dict,
        agent_components: dict
) -> optim.Optimizer:
    """Configures optimizer with specific parameter groups and learning rates."""
    opt_cfg = config_dict['hyper_params']['optimizer']
    opt_cls = getattr(optim, opt_cfg['type'])

    opt_params = opt_cfg.get('params', {})
    lrs_cfg = opt_params.get('lrs', {})
    default_lr = lrs_cfg.get('default', 1e-3)

    # Map all modules and sub-modules for parameter grouping
    module_dict = {}
    for comp_name, comp in agent_components.items():
        if isinstance(comp, nn.Module):
            module_dict[comp_name] = comp
            for child_name, child_module in comp.named_modules():
                if child_name:
                    module_dict[child_name] = child_module

    param_groups = []
    seen_params = set()

    # Assign specific learning rates to targeted modules
    for target_name, lr in lrs_cfg.items():
        if target_name != 'default' and target_name in module_dict:
            target_module = module_dict[target_name]
            params_to_add = [p for p in target_module.parameters() if p not in seen_params]
            if params_to_add:
                param_groups.append({
                    'params': params_to_add,
                    'lr': lr
                })
                seen_params.update(params_to_add)

    # Collect remaining parameters for default learning rate
    remaining_params = []
    for comp in agent_components.values():
        if isinstance(comp, nn.Module):
            remaining_params.extend(
                [p for p in comp.parameters() if p not in seen_params]
            )

    if remaining_params:
        param_groups.append({
            'params': remaining_params,
            'lr': default_lr
        })

    base_params = {k: v for k, v in opt_params.items() if k != 'lrs'}

    return opt_cls(
        param_groups,
        **base_params
    )

def get_criterion(name: str):
    """Retrieves a loss function by name from torch namespaces."""
    if not name: return None
    if hasattr(torch.nn.functional, name):
        return getattr(torch.nn.functional, name)
    if hasattr(torch, name):
        return getattr(torch, name)
    cls = getattr(nn, name, None)
    return cls() if cls else None

def build_component(
        cfg: dict,
        input_dim: int = None
) -> any:
    """Recursively builds components from registry instead of globals."""
    cls_type = cfg.get('type')
    # Safe lookup from registry
    cls = COMPONENT_REGISTRY.get(cls_type)

    if cls is None:
        raise ValueError(f"Class {cls_type} not found in COMPONENT_REGISTRY.")

    # Type A: Components with internal networks
    if 'layers' in cfg:
        in_dim = cfg.get('input_dim', input_dim)
        net, out_dim = build_sequential(
            cfg['layers'],
            in_dim
        )

        if cls == nn.Sequential:
            return net

        params = cfg.get('params', {}).copy()
        if 'feature_dim' not in params:
            params['feature_dim'] = out_dim

        return cls(
            network=net,
            **params
        )

    # Type B: Components containing sub-components
    elif 'components' in cfg:
        sub_components = {}
        feat_dim = cfg.get('feature_dim', input_dim)

        for sub_name, sub_cfg in cfg['components'].items():
            if isinstance(sub_cfg, list):
                sub_net, _ = build_sequential(
                    sub_cfg,
                    feat_dim
                )
                sub_components[sub_name] = sub_net
            else:
                sub_components[sub_name] = build_component(
                    sub_cfg,
                    input_dim=feat_dim
                )

        params = cfg.get('params', {}).copy()
        return cls(
            **sub_components,
            **params
        )

    # Type C: Basic Components
    else:
        params = cfg.get('params', {}).copy()
        return cls(**params)

def build_agent(
        config_dict: dict,
        device: str = 'cpu'
) -> any:
    """Orchestrates the full agent construction process."""
    # 1. Initialize components from state dimension
    state_dim = config_dict['env']['state_dim']
    components = {}

    for name, cfg in config_dict['agent']['components'].items():
        components[name] = build_component(
            cfg,
            input_dim=state_dim
        )

    # 2. Setup optimizer and handle hyper-parameters
    trainable_modules = {
        k: v for k, v in components.items() if isinstance(v, nn.Module)
    }
    optimizer = build_optimizer(
        config_dict,
        trainable_modules
    )

    hyper_params = config_dict.get('hyper_params', {}).copy()
    if 'optimizer' in hyper_params:
        del hyper_params['optimizer']

    if 'criterion' in hyper_params:
        criteria_cfg = hyper_params.pop('criterion')
        for key, loss_name in criteria_cfg.items():
            if loss_name:
                loss_cls = getattr(nn, loss_name)
                hyper_params[key] = loss_cls()

    # 3. Assemble and return the final agent
    agent_type = config_dict['agent']['type']
    agent_cls = COMPONENT_REGISTRY.get(agent_type)

    if agent_cls is None:
        raise ValueError(f"Agent class {agent_type} not registered.")

    instance = agent_cls(
        **components,
        optimizer=optimizer,
        **hyper_params
    )

    instance.to(torch.device(device))
    logger.info(instance.summary())
    return instance
