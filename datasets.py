import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
import json
from tqdm import tqdm


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class protocol_filter:
    def __init__(self, protocols: List):
        self.protocols = protocols

    def __call__(self, RawDataSample):
        if RawDataSample.metadata['acquisition'] in self.protocols:
            return True
        else:
            return False


def get_mgrid_normed(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    if dim != 2:
        raise ValueError("The dim should be 2. Get {}", format(dim))
    tensors = tuple([torch.linspace(-1, 1, steps=sidelen), torch.linspace(1, -1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_mgrid(sidelen, dim=2):
    if dim != 2:
        raise ValueError("The dim should be 2. Get {}", format(dim))
    arrays = tuple([np.arange(sidelen), np.arange(sidelen)])
    meshgrid = np.meshgrid(*arrays, indexing='ij')
    #smeshgrid = [np.zeros((320,320)),np.zeros((320,320))]
    mgrid = np.stack(meshgrid, axis=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def positional_encoding_2d(input_tensor, num_freqs=128):
    # Extract x and y coordinates
    x = input_tensor[:, 0]
    y = input_tensor[:, 1]

    # Calculate frequencies for positional encoding
    frequencies_x = torch.pow(2.0, torch.arange(0, num_freqs).float() / 2.0)
    frequencies_y = torch.pow(2.0, (torch.arange(0, num_freqs).float() + 1.0) / 2.0)

    # Compute positional encoding for x and y coordinates
    encoding_x = torch.sin(frequencies_x * x.unsqueeze(-1))
    encoding_y = torch.cos(frequencies_y * y.unsqueeze(-1))

    # Concatenate the positional encodings along the feature dimension
    encoding = torch.cat([encoding_x, encoding_y], dim=-1)

    return encoding


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg, device=x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(*shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def image_to_patches(image_tensor, x, y):
    if image_tensor.dim() != 3:
        raise ValueError("The shape of image_tensor should be CHW")
    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Calculate padding
    pad_h = (x - 1) // 2
    pad_w = (y - 1) // 2
    padding = (pad_w, pad_w, pad_h, pad_h)

    # Apply padding
    image_tensor_padded = torch.nn.functional.pad(image_tensor, padding, mode='reflect')

    # Use unfold to extract sliding local blocks of a x b
    patches = image_tensor_padded.unfold(2, x, 1).unfold(3, y, 1)

    # Reshape the patches to flatten each patch into a vector
    b, c, h, w, _, _ = patches.size()
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c * x * y)

    return patches


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class IXIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        paths: list,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        # num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        self.raw_samples = []

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = paths
            for fname in sorted(files):
                fname = Path(fname)
                metadata, num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                for slice_ind in range(num_slices//4, num_slices//4*3):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    # if self.raw_sample_filter(raw_sample):
                    #     new_raw_samples.append(raw_sample)
                    new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                print(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            print(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            recon = hf["reconstruction_rss"][dataslice]
            # mask = np.asarray(hf["mask"]) if "mask" in hf else None

            # target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

            if self.transform is None:
                raise ValueError('Transform should not be None.')
                # sample = (kspace, mask, target, attrs, fname.name, dataslice)
            else:
                # sample = self.transform(kspace, target, attrs, fname.name, dataslice)
                sample = self.transform(kspace, recon, attrs, fname.name, dataslice)
        return sample


class IXISliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        paths: list,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "ixi_dataset_cache.pkl",
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        self.raw_samples = []

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = paths
            for fname in sorted(files):
                fname = Path(fname)
                num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                begin = num_slices // 4
                end = num_slices // 4 * 3
                for slice_ind in range(begin, end):
                    raw_sample = IXIRawDataSample(fname, slice_ind)
                    # if self.raw_sample_filter(raw_sample):
                    #     new_raw_samples.append(raw_sample)
                    new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                print(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            print(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

    def _retrieve_metadata(self, fname):
        image_sitk = sitk.ReadImage(fname)
        image_array_sitk = sitk.GetArrayFromImage(image_sitk)
        num_slices = image_array_sitk.shape[0]
        return num_slices

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice = self.raw_samples[i]
        image_sitk = sitk.ReadImage(fname)
        image_array_sitk = sitk.GetArrayFromImage(image_sitk)
        gt = np.array(image_array_sitk[dataslice])
        if self.transform is None:
            raise ValueError('Transform should not be None.')
        else:
            sample = self.transform(gt, fname.name, dataslice)
        return sample


class SimpleSliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        protocol: str,
        recon_size: int,
        pos_encoding_num_freqs: int,
        split: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        # num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        if protocol not in ("pd", "pdfs"):
            raise ValueError('protocol should be either "pd" or "pdfs"')  # TODO: Need to add the protocols of brain

        if split not in ("train", "val", "test"):
            raise ValueError('split should be among "train", "val" and "test"')

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            # data_files = list((Path(root) / protocol / (challenge + "_" + split) / "data").iterdir())
            metadata_files = list((Path(root) / protocol / (challenge + "_" + split) / "metadata").iterdir())
            with tqdm(total=len(metadata_files)) as pbar:
                pbar.set_description('Making dataset cache')
                for fname in sorted(metadata_files):
                    with open(fname, 'r') as jf:
                        metadata = json.load(jf)
                        fname_without_extension = fname.stem
                        data_fname = Path(root) / protocol / (challenge + "_" + split) / "data" / (fname_without_extension + '.npy')
                        num_slices = np.load(data_fname).shape[0]
                    new_raw_samples = []
                    for slice_ind in range(num_slices):
                        raw_sample = FastMRIRawDataSample(data_fname, slice_ind, metadata)
                        if self.raw_sample_filter(raw_sample):
                            new_raw_samples.append(raw_sample)
                    self.raw_samples += new_raw_samples
                    pbar.update(1)

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                print(f"Saving dataset cache to {self.dataset_cache_file}.")
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            print(f"Using dataset cache from {self.dataset_cache_file}.")
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        # To improve the efficiency, we prepare the coords and coords_norm first for only once assuming that all images
        # are supposed to be reconstructed to a same square size.
        self.coords = get_mgrid(recon_size, 2)
        self.coords_normed = get_mgrid_normed(recon_size, 2)
        self.pos_encoding = positional_encoding_2d(self.coords_normed, num_freqs=pos_encoding_num_freqs)
        # if num_cols:
        #     self.raw_samples = [
        #         ex
        #         for ex in self.raw_samples
        #         if ex[2]["encoding_size"][1] in num_cols  # type: ignore
        #     ]

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]

        kspace = np.load(fname)[dataslice]
        if self.transform is None:
            raise ValueError('Transform should not be None.')
            # sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            # sample = self.transform(kspace, target, attrs, fname.name, dataslice)
            sample = self.transform(kspace, metadata, fname.name, dataslice)
        return sample


class ImageFitting(torch.utils.data.Dataset):
    def __init__(self, sidelength, sample, pos_dim=128):
        super().__init__()
        self.d_pixels = sample.image[None].permute(1, 2, 0).view(-1, 1)
        self.pixels = sample.target[None].permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid_normed(sidelength, 2)
        self.d_image = sample.image[None]
        self.gt_image = sample.target[None]
        self.mask = sample.mask[None]
        # self.pos_encoding = positional_encoding_2d(self.coords, pos_dim)
        self.pos_encoding = pos_enc(self.coords, min_deg=0, max_deg=pos_dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0 : raise IndexError

        return self.coords, self.pos_encoding, self.d_pixels, self.pixels, self.d_image, self.gt_image, self.mask


class ImageFittingPG(torch.utils.data.Dataset):
    def __init__(self, sidelength, sample, pos_dim=128):
        super().__init__()
        self.d_pixels = sample.image[None].permute(1, 2, 0).view(-1, 1)
        self.pixels = sample.target[None].permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid_normed(sidelength, 2)
        self.d_image = sample.image[None]
        self.gt_image = sample.target[None]
        self.under_mask = sample.under_mask
        self.select_mask = sample.select_mask
        self.remain_mask = sample.remain_mask
        # self.pos_encoding = positional_encoding_2d(self.coords, pos_dim)
        self.pos_encoding = pos_enc(self.coords, min_deg=0, max_deg=pos_dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0 : raise IndexError

        return (self.coords, self.pos_encoding, self.d_pixels, self.pixels, self.d_image, self.gt_image, self.under_mask,
                self.remain_mask, self.select_mask)


class IXIImageFitting(torch.utils.data.Dataset):
    def __init__(self, sidelength, sample, pos_dim=128):
        super().__init__()
        self.d_pixels = sample.image[None].permute(1, 2, 0).view(-1, 1)
        self.pixels = sample.target[None].permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid_normed(sidelength, 2)
        self.d_image = sample.image[None]
        self.gt_image = sample.target[None]
        self.mask = sample.mask[None]
        self.pos_encoding = pos_enc(self.coords, min_deg=0, max_deg=pos_dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0 : raise IndexError

        return self.coords, self.pos_encoding, self.d_pixels, self.pixels, self.d_image, self.gt_image, self.mask


class IXIImageFittingPG(torch.utils.data.Dataset):
    def __init__(self, sidelength, sample, pos_dim=128):
        super().__init__()
        self.d_pixels = sample.image[None].permute(1, 2, 0).view(-1, 1)
        self.pixels = sample.target[None].permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid_normed(sidelength, 2)
        self.d_image = sample.image[None]
        self.gt_image = sample.target[None]
        self.under_mask = sample.under_mask
        self.select_mask = sample.select_mask[None]
        self.remain_mask = sample.remain_mask[None]
        self.pos_encoding = pos_enc(self.coords, min_deg=0, max_deg=pos_dim)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0 : raise IndexError

        return (self.coords, self.pos_encoding, self.d_pixels, self.pixels, self.d_image, self.gt_image, self.under_mask,
                self.remain_mask, self.select_mask)


class DIPUNetSliceDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        super().__init__()
        self.img = sample.image
        self.lb = sample.target
        self.mask = sample.mask
        self.noise = self.get_noise(32, 'noise', (320, 320)).view(32, 320, 320)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0 : raise IndexError

        return self.img, self.lb, self.mask, self.noise

    def fill_noise(self, x, noise_type):
        """Fills tensor `x` with noise of type `noise_type`."""
        if noise_type == 'u':
            x.uniform_()
        elif noise_type == 'n':
            x.normal_()
        else:
            assert False

    def np_to_torch(self, img_np):
        '''Converts image in numpy.array to torch.Tensor.

        From C x W x H [0..1] to  C x W x H [0..1]
        '''
        return torch.from_numpy(img_np)[None, :]

    def get_noise(self, input_depth, method, spatial_size, noise_type='u', var=1. / 10):
        """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
        initialized in a specific way.
        Args:
            input_depth: number of channels in the tensor
            method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
            spatial_size: spatial size of the tensor to initialize
            noise_type: 'u' for uniform; 'n' for normal
            var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
        """
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        if method == 'noise':
            shape = [1, input_depth, spatial_size[0], spatial_size[1]]
            net_input = torch.zeros(shape)

            self.fill_noise(net_input, noise_type)
            net_input *= var
        elif method == 'meshgrid':
            assert input_depth == 2
            X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                               np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
            meshgrid = np.concatenate([X[None, :], Y[None, :]])
            net_input = self.np_to_torch(meshgrid)
        else:
            assert False

        return net_input


def create_data_loader(dataset, batch_size, num_workers, sampler=None, is_train=True):
    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )
    return dataloader
