"""Shared image access for the image-only and fusion models (PROJET.md §5).

Images are the input of src/vit.py and src/clip_fusion.py. They are read from
the paths stored in the cleaned splits, relative to the repository root, and
returned in the row order of the split so that predictions stay aligned with
the population src/evaluate.py validates. Nothing is resized or normalized
here: each model's own processor owns that. A missing or unreadable file raises,
never silently drops a row, because the evaluation population is fixed by
src/data.py and src/fetch_images.py has already validated every file.
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def open_rgb(image, root=ROOT):
    """Fully decoded RGB image for one split row; raises if the file is unusable."""
    path = Path(root) / image
    try:
        with Image.open(path) as handle:
            # convert() decodes the pixels, so a truncated file fails here and not
            # later inside a training batch.
            return handle.convert('RGB')
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Image referenced by the splits is missing: {path}') from exc
    except (OSError, SyntaxError) as exc:
        raise OSError(f'Unreadable image: {path}: {type(exc).__name__}: {exc}') from exc


def unique_images(*frames):
    """Sorted unique image paths of one or more split DataFrames."""
    paths = set()
    for frame in frames:
        paths.update(frame['image'])
    return sorted(paths)


class ImageDataset(Dataset):
    """Preprocessed pixel values, and class ids when given, in the order received.

    ``images`` is a sequence of split-relative paths: pass ``frame['image']``
    to keep the DataFrame row order. ``transform`` maps a PIL image to the
    tensor the model expects.
    """

    def __init__(self, images, transform, labels=None, root=ROOT):
        self.images = list(images)
        self.labels = None if labels is None else list(labels)
        if self.labels is not None and len(self.labels) != len(self.images):
            raise ValueError('One label is required per image')
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = {'pixel_values': self.transform(open_rgb(self.images[index], self.root))}
        if self.labels is not None:
            item['labels'] = self.labels[index]
        return item
