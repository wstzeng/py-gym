import copy

import gymnasium as gym
import torch
from torch import nn, optim

import agent
import agent.components.buffer
import agent.components.encoder
import agent.components.policy
import agent.components.policy.action_handler
from utils.logger import logger


def get_component_registry() -> dict:
    registry = {
        "Sequential": nn.Sequential,
    }

    modules = [
        agent,
        agent.components.encoder,
        agent.components.policy,
        agent.components.policy.action_handler,
        agent.components.buffer,
    ]

    for mod in modules:
        if hasattr(mod, "__all__"):
            for name in mod.__all__:
                registry[name] = getattr(mod, name)
        else:
            logger.warning(f"Module {mod.__name__} has no __all__ defined.")

    return registry


COMPONENT_REGISTRY = get_component_registry()


def build_sequential(
        layer_configs: list,
        input_dim: int = None
) -> tuple[nn.Module, int]:
    """Build a sequential network and return its output dimension."""
    layers = []
    current_dim = input_dim

    for cfg in layer_configs:
        layer_type = getattr(nn, cfg['type'])
        params = {k: v for k, v in cfg.items() if k != 'type'}
        params.pop('in_features', None)

        if layer_type is nn.Linear:
            if current_dim is None:
                raise ValueError(
                    "Cannot infer input dimension for Linear layer. "
                    "Provide input_dim on the component or its parent."
                )
            layer = layer_type(in_features=current_dim, **params)
            current_dim = params['out_features']
        else:
            layer = layer_type(**params)

        layers.append(layer)

    return nn.Sequential(*layers), current_dim


def build_optimizer(
        optimizer_cfg: dict,
        agent_components: dict
) -> optim.Optimizer:
    """Build an optimizer with optional per-module learning rates."""
    opt_cls = getattr(optim, optimizer_cfg['type'])
    opt_params = optimizer_cfg.get('params', {})
    lrs_cfg = opt_params.get('lrs', {})
    default_lr = lrs_cfg.get('default', 1e-3)

    # Map module names to modules so lrs can target nested components.
    module_dict = {}
    for comp_name, comp in agent_components.items():
        if isinstance(comp, nn.Module):
            module_dict[comp_name] = comp
            for child_name, child_module in comp.named_modules():
                if child_name:
                    module_dict[child_name] = child_module

    param_groups = []
    seen_params = set()

    for target_name, lr in lrs_cfg.items():
        if target_name == 'default' or target_name not in module_dict:
            continue

        target_module = module_dict[target_name]
        params_to_add = [
            p for p in target_module.parameters()
            if p not in seen_params
        ]
        if params_to_add:
            param_groups.append({
                'params': params_to_add,
                'lr': lr,
            })
            seen_params.update(params_to_add)

    remaining_params = []
    for comp in agent_components.values():
        if isinstance(comp, nn.Module):
            remaining_params.extend(
                p for p in comp.parameters()
                if p not in seen_params
            )

    if remaining_params:
        param_groups.append({
            'params': remaining_params,
            'lr': default_lr,
        })

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    base_params = {
        k: v for k, v in opt_params.items()
        if k != 'lrs'
    }
    return opt_cls(param_groups, **base_params)


def build_component(
        cfg: dict,
        input_dim: int = None
) -> tuple[object, int]:
    """
    Recursively build a component from config.

    Returns the component and its output dimension when it can be inferred.
    Sub-components inherit ``input_dim`` unless they provide their own
    ``input_dim``; this lets encoder -> policy -> buffer dimensions flow
    through the config without repeating them.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"Component config must be a dict, got: {type(cfg)}")

    cls_type = cfg.get('type')
    cls = COMPONENT_REGISTRY.get(cls_type)

    if cls is None:
        raise ValueError(
            f"Class {cls_type!r} not found in COMPONENT_REGISTRY. "
            f"Export it in the package __all__."
        )

    params = dict(cfg.get('params', {}))

    # Components containing named sub-components.
    if hasattr(cls, 'sub_components'):
        sub_components = {}
        for sub_name in cls.sub_components():
            sub_cfg = cfg.get(sub_name)
            if sub_cfg is None:
                raise ValueError(
                    f"{cls_type} requires sub-component {sub_name!r}."
                )
            sub_components[sub_name], _ = build_component(
                sub_cfg,
                input_dim=input_dim,
            )

        instance = cls(**sub_components, **params)
        return instance, getattr(instance, 'feature_dim', None)

    # Components with an internal sequential network.
    if 'layers' in cfg:
        in_dim = cfg.get('input_dim', input_dim)
        network, out_dim = build_sequential(cfg['layers'], in_dim)

        if cls is nn.Sequential:
            return network, out_dim

        params.setdefault('feature_dim', cfg.get('feature_dim', out_dim))
        instance = cls(network=network, **params)
        return instance, getattr(instance, 'feature_dim', out_dim)

    # Basic components such as buffers and action handlers.
    if getattr(cls, 'requires_input_dim', False):
        if input_dim is None:
            raise ValueError(
                f"{cls_type} requires an input dimension, but none was "
                "provided by its parent component."
            )
        params.setdefault('input_dim', input_dim)

    instance = cls(**params)
    return instance, getattr(instance, 'feature_dim', None)


def build_agent(
        config_dict: dict,
        device: str = 'cpu'
) -> object:
    """Build a complete agent from a parsed configuration dictionary."""
    config = copy.deepcopy(config_dict)
    env_cfg = config['env']

    env = gym.make(env_cfg['id'], **env_cfg.get('default', {}))
    try:
        if env.observation_space.shape is None:
            raise ValueError(
                f"Cannot infer state dimension from {env_cfg['id']}."
            )
        state_dim = env.observation_space.shape[0]
    finally:
        env.close()

    agent_cfg = config['agent']
    agent_type = agent_cfg.pop('type', None)
    agent_cls = COMPONENT_REGISTRY.get(agent_type)

    if agent_cls is None:
        raise ValueError(
            f"Agent class {agent_type!r} not found in COMPONENT_REGISTRY."
        )

    component_names = list(agent_cls.sub_components())
    unknown_keys = set(agent_cfg) - set(component_names)
    if unknown_keys:
        raise ValueError(
            f"Unknown agent config keys: {sorted(unknown_keys)}. "
            f"Expected: {component_names}"
        )

    components = {}
    input_dim = state_dim
    for name in component_names:
        if name not in agent_cfg:
            raise ValueError(
                f"{agent_type} requires agent component {name!r}."
            )

        component, output_dim = build_component(
            agent_cfg[name],
            input_dim=input_dim,
        )
        components[name] = component

        if output_dim is not None:
            input_dim = output_dim

    hyper_params = dict(config.get('hyper_params', {}))
    optimizer_cfg = hyper_params.pop('optimizer', None)
    if optimizer_cfg is None:
        raise ValueError("Missing hyper_params.optimizer config.")

    criteria_cfg = hyper_params.pop('criterion', {})
    criteria = {}
    for key, loss_name in criteria_cfg.items():
        if not loss_name:
            continue
        loss_cls = getattr(nn, loss_name, None)
        if loss_cls is None:
            raise ValueError(f"Unknown loss function: {loss_name!r}")
        criteria[key] = loss_cls()

    optimizer = build_optimizer(optimizer_cfg, components)

    instance = agent_cls(
        **components,
        optimizer=optimizer,
        hyper_params=hyper_params,
        **criteria,
    )
    instance.to(torch.device(device))
    logger.info(instance.summary())
    return instance
