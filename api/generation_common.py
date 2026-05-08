"""Shared helpers for the generation runtime."""


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def _get_first(config, names, default=None):
    for name in names:
        if name in config:
            return config[name]
    return default


def _option_enabled(config):
    return bool(config.get('enabled', config.get('merge', False)))


def _as_bool(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'true', '1', 'yes', 'y', 'on'}:
            return True
        if value in {'false', '0', 'no', 'n', 'off'}:
            return False
        raise ValueError(f'Cannot parse boolean value: {value!r}')
    return bool(value)


def _template_uses_capping(template):
    if str(template.get('class', '')) != 'Peptide':
        return True
    return _as_bool(template.get('capping', True))


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
