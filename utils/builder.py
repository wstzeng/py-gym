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
    """Builds optimizer and ensures no parameter is added twice."""
    opt_cfg = config_dict['optimizer']
    opt_cls = getattr(optim, opt_cfg['type'])

    custom_lrs = config_dict['hyper_params'].get('custom_lrs', {})
    default_lr = opt_cfg['params'].get('lr', 1e-3)

    param_groups = []
    seen_params = set()

    for comp_name, module in agent_components.items():
        lr = custom_lrs.get(comp_name, default_lr)
        
        params_to_add = []
        for p in module.parameters():
            if p not in seen_params:
                params_to_add.append(p)
                seen_params.add(p)
        
        if params_to_add:
            param_groups.append({'params': params_to_add, 'lr': lr})

    base_params = {k: v for k, v in opt_cfg['params'].items() if k != 'lr'}
    return opt_cls(param_groups, **base_params)

def build_agent(
        config_dict: dict,
        device: str = 'cpu'
) -> BaseAgent:
    components = {}
    state_dim = config_dict.get('state_dim')

    for name, cfg in config_dict['components'].items():
        cls_type = cfg['type']
        
        cls = globals().get(cls_type)
        if cls is None:
            raise ValueError(f"Class {cls_type} not found in globals.")

        # --- Scenario A: Encoder/Simple Networks ---
        if 'layers' in cfg:
            input_dim = cfg.get('input_dim', state_dim)
            net, out_dim = build_sequential(cfg['layers'], input_dim)
            
            params = cfg.get('params', {})
            if 'feature_dim' not in params:
                params['feature_dim'] = out_dim
                
            components[name] = cls(network=net, **params)

        # --- Scenario B: Complex Policies (Actor-Critic / Actor) ---
        elif 'sub_networks' in cfg:
            sub_nets = {}
            feat_dim = cfg.get('feature_dim')
            
            for sub_name, layers in cfg['sub_networks'].items():
                sub_nets[sub_name], _ = build_sequential(layers, feat_dim)
            
            params = cfg.get('params', {}).copy()
            for p_name, p_val in params.items():
                if isinstance(p_val, str) and p_val in components:
                    params[p_name] = components[p_val]

            components[name] = cls(**sub_nets, **params)

        # --- Scenario C & D: Distributor / Buffer / Simple Classes ---
        else:
            components[name] = cls(**cfg.get('params', {}))

    trainable_modules = {k: v for k, v in components.items() if isinstance(v, nn.Module)}
    optimizer = build_optimizer(config_dict, trainable_modules)

    agent_cfg = config_dict['agent']
    agent_cls = globals()[agent_cfg['type']]
    agent_params = {k: v for k, v in config_dict['hyper_params'].items() if k != 'custom_lrs'}

    return agent_cls(
        **components,
        optimizer=optimizer,
        device=device,
        **agent_params
    )
