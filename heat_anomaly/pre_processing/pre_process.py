
import logging
from typing import Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2

from heat_anomaly.data.utils import get_image_height_and_width

logger = logging.getLogger(__name__)


def get_transforms(
    config: Optional[Union[str, A.Compose]] = None,
    image_size: Optional[Union[int, Tuple]] = None,
    to_tensor: bool = True,
) -> A.Compose:
    
    if config is None and image_size is None:
        raise ValueError(
            "Both config and image_size cannot be `None`. "
            "Provide either config file to de-serialize transforms "
            "or image_size to get the default transformations"
        )

    transforms: A.Compose

    if config is None and image_size is not None:
        height, width = get_image_height_and_width(image_size)
        transforms = A.Compose(
            [
                A.Resize(height=height, width=width, always_apply=True),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ]
        )

    if config is not None:
        if isinstance(config, str):
            transforms = A.load(filepath=config, data_format="yaml")
        elif isinstance(config, A.Compose):
            transforms = config
        else:
            raise ValueError("config could be either ``str`` or ``A.Compose``")

    if not to_tensor:
        if isinstance(transforms[-1], ToTensorV2):
            transforms = A.Compose(transforms[:-1])

    # always resize to specified image size
    if not any(isinstance(transform, A.Resize) for transform in transforms) and image_size is not None:
        height, width = get_image_height_and_width(image_size)
        transforms = A.Compose([A.Resize(height=height, width=width, always_apply=True), transforms])

    return transforms


class PreProcessor:

    def __init__(
        self,
        config: Optional[Union[str, A.Compose]] = None,
        image_size: Optional[Union[int, Tuple]] = None,
        to_tensor: bool = True,
    ) -> None:
        self.config = config
        self.image_size = image_size
        self.to_tensor = to_tensor

        self.transforms = get_transforms(config, image_size, to_tensor)

    def __call__(self, *args, **kwargs):
        """Return transformed arguments."""
        return self.transforms(*args, **kwargs)
