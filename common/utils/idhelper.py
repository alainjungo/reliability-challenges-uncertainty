import time


def get_unique_identifier() -> str:
    return time.strftime('%y%m%d-%H%M%S')


def extract_leading_identifier(value: str) -> str:
    if len(value) < 13:
        return ''
    time_str = value[:13]
    try:
        time.strptime(time_str, '%y%m%d-%H%M%S')
    except ValueError:
        return ''
    return time_str
