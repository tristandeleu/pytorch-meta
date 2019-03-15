import os
import json

def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)

def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError()

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data

def download_google_drive(id, filepath):
    import requests
    from tqdm import tqdm
    url = 'https://docs.google.com/uc?export=download'

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination, chunksize=32768):
        with open(destination, "wb") as f:
            with tqdm(desc=filename, unit_scale=True, unit_divisor=1024, unit='b') as pbar:
                for chunk in response.iter_content(chunksize):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    session = requests.Session()
    params = {'id': id}
    response = session.get(url, params=params, stream=True)
    token = get_confirm_token(response)

    if token is not None:
        params.update({'confirm': token})
        response = session.get(url, params=params, stream=True)

    save_response_content(response, filepath)
