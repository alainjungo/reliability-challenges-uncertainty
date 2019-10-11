

def get_type_error_msg(obj, *expected_cls):
    if len(expected_cls) > 1:
        exp = '({})'.format(','.join(e.__name__ for e in expected_cls))
    else:
        exp = expected_cls[0].__name__

    return 'expected type is "{}" but object is of type "{}"'.format(exp, obj.__class__.__name__)
