import os
import json


def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data

# QKFIX: The current version of `download_file_from_google_drive` (as of torchvision==0.8.1)
# is inconsistent, and a temporary fix has been added to the bleeding-edge version of
# Torchvision. The temporary fix removes the behaviour of `_quota_exceeded`, whenever the
# quota has exceeded for the file to be downloaded. As a consequence, this means that there
# is currently no protection against exceeded quotas. If you get an integrity error in Torchmeta
# (e.g. "MiniImagenet integrity check failed" for MiniImagenet), then this means that the quota
# has exceeded for this dataset. See also: https://github.com/tristandeleu/pytorch-meta/issues/54
# 
# See also: https://github.com/pytorch/vision/issues/2992
# 
# The following functions are taken from
# https://github.com/pytorch/vision/blob/cd0268cd408d19d91f870e36fdffd031085abe13/torchvision/datasets/utils.py

from torchvision.datasets.utils import _get_confirm_token, _save_response_content

def _quota_exceeded(response: "requests.models.Response"):
    return False
    # See https://github.com/pytorch/vision/issues/2992 for details
    # return "Google Drive - Quota exceeded" in response.text


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        if _quota_exceeded(response):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(response, fpath)
