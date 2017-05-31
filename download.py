import os
import sys
import urllib.request
import zipfile


def maybe_download_from_url(url, download_dir):
    """
    Download the data from url, unless it's already here.

    Args:
        download_dir: string, path to download directory
        url: url to download from

    Returns:
        Path to the downloaded file
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(download_dir, filename)

    os.makedirs(download_dir, exist_ok=True)

    if not os.path.isfile(filepath):
        print('Downloading: "{}"'.format(filepath))
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print('Download complete ({} bytes)'.format(size))
    else:
        print('File already exists: "{}"'.format(filepath))

    return filepath


def maybe_extract(compressed_filepath, target_dir):
    def is_image(filepath):
        extensions = ('.jpg', '.jpeg', '.png', '.gif')
        return any(filepath.endswith(ext) for ext in extensions)

    os.makedirs(target_dir, exist_ok=True)
    print('Extracting: "{}"'.format(compressed_filepath))

    if zipfile.is_zipfile(compressed_filepath):
        with zipfile.ZipFile(compressed_filepath) as zf:
            files = [member for member in zf.infolist() if is_image(member.filename)]
            for file in files:
                if not os.path.exists(os.path.join(target_dir, file.filename)):
                    zf.extract(file, target_dir)
    else:
        raise NotImplemented

    print('Extraction complete')
