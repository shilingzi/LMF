import copy


# 模型注册中心
_model_factory = {}


def register(name):
    def register_model_class(cls):
        if name in _model_factory:
            raise ValueError(f"已存在名为 '{name}' 的模型")
        _model_factory[name] = cls
        return cls
    return register_model_class


def make(model_spec, **kwargs):
    """
    创建指定模型的实例

    Args:
        model_spec (dict): 模型规格
        **kwargs: 其他参数

    Returns:
        模型实例
    """
    if isinstance(model_spec, str):
        model_name = model_spec
        model_args = kwargs
    elif isinstance(model_spec, dict):
        model_name = model_spec['name']
        model_args = model_spec.get('args', {})
        model_args.update(kwargs)
    else:
        raise ValueError('模型规格必须是字符串或字典')
        
    if model_name not in _model_factory:
        raise ValueError(f"未知模型: '{model_name}'")
        
    return _model_factory[model_name](**model_args) 