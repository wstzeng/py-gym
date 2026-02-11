import torch
from torch import nn, optim
from agent.encoder import *
from agent.policy import *
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
    """Builds optimizer with support for parameter groups (custom LRs)."""
    opt_cfg = config_dict['optimizer']
    opt_cls = getattr(optim, opt_cfg['type'])
    
    custom_lrs = config_dict['hyper_params'].get('custom_lrs', {})
    default_lr = opt_cfg['params'].get('lr', 1e-3)
    
    param_groups = []
    for comp_name, module in agent_components.items():
        # Scans internal children (e.g., policy.actor, policy.critic)
        for name, child in module.named_children():
            lr = custom_lrs.get(name, default_lr)
            param_groups.append({'params': child.parameters(), 'lr': lr})
            
    # Remove default lr from params to avoid override if needed
    base_params = {k: v for k, v in opt_cfg['params'].items() if k != 'lr'}
    return opt_cls(param_groups, **base_params)

def build_agent(
        config_dict: dict,
        device: str = 'cpu'
) -> BaseAgent:
    """
    Dependency Injection Container:
    Maps JSON blueprint to live objects and injects them into the Agent.
    """
    components = {}
    state_dim = config_dict.get('state_dim')

    for name, cfg in config_dict['components'].items():
        cls = globals()[cfg['type']]

        # Scenario A: Component needs an internal neural network (e.g., Encoder)
        if 'layers' in cfg:
            input_dim = cfg.get('input_dim', config_dict.get('state_dim'))
            net, feature_dim = build_sequential(
                layer_configs=cfg['layers'], 
                input_dim=input_dim
            )
            cls = globals()[cfg['type']]

            # Ensure feature_dim is passed to the Encoder constructor
            encoder_params = cfg.get('params', {})
            if 'feature_dim' not in encoder_params:
                encoder_params['feature_dim'] = feature_dim

            components[name] = cls(
                network=net, 
                **encoder_params
            )

        # Scenario B: Component has sub-networks (e.g., Actor-Critic Policy)
        elif 'sub_networks' in cfg:
            sub_nets = {}
            for sub_name, layers in cfg['sub_networks'].items():
                # Note: Here we might need a way to pass 'feature_dim' correctly
                # Simplified: assuming sub_networks take 'input_dim' from config
                feat_dim = cfg.get('feature_dim', state_dim) 
                sub_nets[sub_name], _ = build_sequential(layers, feat_dim)
            components[name] = cls(**sub_nets, **cfg.get('params', {}))
            
        # Scenario C: Simple class (e.g., Buffer)
        else:
            components[name] = cls(**cfg.get('params', {}))

    # Identify modules that require training
    trainable_modules = {k: v for k, v in components.items() if isinstance(v, nn.Module)}
    optimizer = build_optimizer(config_dict, trainable_modules)

    # Initialize Agent by injecting all built components
    agent_cfg = config_dict['agent']
    agent_cls = globals()[agent_cfg['type']]
    
    # hyper_params filtered to exclude builder-only keys like 'custom_lrs'
    agent_params = {k: v for k, v in config_dict['hyper_params'].items() if k != 'custom_lrs'}

    return agent_cls(
        **components,
        optimizer=optimizer,
        device=device,
        **agent_params
    )
