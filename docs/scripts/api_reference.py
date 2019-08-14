import importlib
import inspect
import re
import yaml
import logging
import torch.nn as nn

LAST_CHARACTER = re.compile(r'^[\w\.,\"\`]$')
HEADER = re.compile(r'^\-+$')
REFERENCE = re.compile(r'^\.\. \[(\d)+\] (.+)$')

def format_signature(string, maxlen=77):
    """Format the signature of the function, with line breaks after
    `maxlen + 3` characters."""
    line, *elements = string.split(', ')
    lines = []
    while elements:
        elem, *elements = elements
        if len(line) + len(elem) < maxlen:
            line = '{0}, {1}'.format(line, elem)
        else:
            lines.append('{0},'.format(line))
            line = '    {0}'.format(elem)
    if line:
        lines.append(line)
    return '```python\n{0}\n```'.format('\n'.join(lines))

def format_parameters(section):
    """Format the "Parameters" section."""
    def format_item(item):
        item = map(lambda x: x.strip(), item)
        return ' - **{0}**: *{1}*\n {2}'.format(*item)
    return '**Parameters**\n\n{0}'.format('\n\n'.join(
        map(format_item, section)))

def format_notes(section):
    """Format the "Notes" section."""
    assert len(section) == 1
    return '!!! note "Notes"\n    {0}'.format(section[0].strip())

def format_references(section):
    """Format the "References" section."""
    def format_item(item):
        return '    - **[{0}]** {1}'.format(item[0], item[1].strip())
    return '!!! attention "References"\n{0}'.format('\n'.join(
        map(format_item, section)))

def format_docs(module, kls):
    """Format the documentation for a module."""
    assert getattr(kls, '__doc__') is not None
    markdown = ['## {0}'.format(kls.__name__)]
    sections = parse_docs(kls.__doc__)
    for header, section in sections:
        if header is None:
            # Add special case for inherited documentation
            if len(sections) > 1:
                markdown.append(section[0].strip())
            signature = '{0}.{1}{2}'.format(module.__name__, kls.__name__,
                str(inspect.signature(kls)))
            markdown.append(format_signature(signature))
            if len(sections) == 1 and any(c is nn.Module
                    for c in inspect.getmro(kls)):
                markdown.append(format_notes(['See: `torch.nn.{0}`'.format(
                    kls.__name__[4:])]))
        elif header == 'Parameters':
            markdown.append(format_parameters(section))
        elif header == 'Notes':
            markdown.append(format_notes(section))
        elif header == 'References':
            markdown.append(format_references(section))
    return '\n\n'.join(markdown)

def merge_lines(line_1, line_2):
    """Merge two consecutive lines, depending on the last character of
    the first line."""
    if not line_1:
        return line_2
    pattern = '{0} {1}' if LAST_CHARACTER.match(line_1[-1]) else '{0}{1}'
    return pattern.format(line_1, line_2)

def parse_docs(docs):
    """Parse the docs as a string into multiple sections."""
    lines = inspect.cleandoc(docs).split('\n')
    sections, section = [], []
    last_header, element = None, None
    for i, line in enumerate(lines):
        if not line:
            line = '\n'
        if HEADER.match(line):
            continue

        if (i < (len(lines) - 1)) and HEADER.match(lines[i + 1]):
            if element is not None:
                section.append(element)
                element = None
            sections.append((last_header, section))
            section = []
            last_header = line
            continue

        if (last_header is None) or (last_header == 'Notes'):
            if not section:
                section = [line]
            else:
                section[0] = merge_lines(section[0], line.strip(' '))

        elif last_header == 'Parameters':
            if line[:4].strip():
                if element is not None:
                    section.append(element)
                element = line.split(':', 1) + ['']
            else:
                element[2] = merge_lines(element[2], line[4:])

        elif last_header == 'References':
            if line[:2] == '..':
                if element is not None:
                    section.append(element)
                match = REFERENCE.match(line)
                element = [int(match[1]), match[2]]
            else:
                element[1] = merge_lines(element[1], line.strip())

    if element is not None:
        section.append(element)
    sections.append((last_header, section))

    return sections

def main(args):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for page in config['api-reference']:
        filename = next(iter(page))
        name, apis = page[filename][0]['name'], page[filename][1]['content']
        content = []
        logging.info('Creating `{0}`...'.format(filename))

        for api in apis:
            module_name = next(iter(api))
            module = importlib.import_module(module_name)
            class_names = api[module_name]

            if (len(class_names) == 1) and (class_names[0] == '__all__'):
                class_names = getattr(module, '__all__')
                if class_names is None:
                    continue

            for class_name in class_names:
                kls = getattr(module, class_name)
                if getattr(kls, '__doc__') is None:
                    continue
                logging.info('  {0}.{1}'.format(module_name, class_name))
                content.append(format_docs(module, kls))

        with open(filename, 'w') as f:
            f.write('\n\n'.join(content))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Torchmeta API reference')
    parser.add_argument('--config', type=str, default='docs/scripts/config.yml',
                        help='Path to the configuration file.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
