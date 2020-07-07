import re
import warnings

from collections import OrderedDict

def get_subdict(dictionary, key=None):
    warnings.warn('The function `torchmeta.modules.utils.get_subdict` is '
                  'deprecated, and will be removed in Torchmeta v1.5. Please '
                  'use the `get_subdict` method from `MetaModule` (e.g. '
                  '`self.get_subdict(params, "{0}")`) instead.'.format(key),
                  stacklevel=2)
    if dictionary is None:
        return None

    if (key is None) or (key == ''):
        return dictionary

    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    # Compatibility with DataParallel
    if not any(filter(key_re.match, dictionary.keys())):
        key_re = re.compile(r'^module\.{0}\.(.+)'.format(re.escape(key)))

    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)
