from pathlib import Path
import natsort

import webdataset as wds

from itertools import islice, cycle



def concatenate_samples(samples, config):
    raise NotImplementedError


def generate_seqs(src, nsamples=3, config=None):
    it = iter(src)
    result = tuple(islice(it, nsamples))
    if len(result) == nsamples:
        yield concatenate_samples(result, config)
    for elem in it:
        result = result[1:] + (elem,)
        yield concatenate_samples(result, config)


def find_tar_files(read_path, pattern):
    files = [str(f) for f in Path(read_path).glob('*.tar') if f.match(pattern + '*')]
    return natsort.natsorted(files)


def get_dataset_paths(config):
    paths = {}
    data_split = config['data_split']
    read_path = config['raw_data_path']
    for key, split in data_split.items():
        combinations = [
            '_'.join(item)
            for item in zip(
                cycle(split['town']), split['season'], cycle(split['behavior'])
            )
        ]

        # Get all the tar files
        temp = [find_tar_files(read_path, combination) for combination in combinations]

        # Concatenate all the paths and assign to dict
        paths[key] = sum(temp, [])  # Not a good way, but it is fun!
    return paths


def webdataset_data_iterator(config):
    # Get the dataset paths
    paths = get_dataset_paths(config)

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    SEQ_LEN = config['obs_size']
    number_workers = config['number_workers']

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}

    for key, path in paths.items():
        if path:
            dataset = (
                wds.WebDataset(path, shardshuffle=False)
                .decode("torchrgb")
                .then(generate_seqs, SEQ_LEN, config)
            )
            data_loader = wds.WebLoader(
                dataset,
                num_workers=number_workers,
                shuffle=False,
                batch_size=BATCH_SIZE,
            )
            if key in ['training', 'validation']:
                dataset_size = 6250 * len(path)
                data_loader.length = dataset_size // BATCH_SIZE

            data_iterator[key] = data_loader

    return data_iterator
