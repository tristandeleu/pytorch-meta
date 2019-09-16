import logging

def main(args):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

    with open('docs/index.md', 'w') as f:
        f.write(long_description)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Torchmeta Doc Index')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
