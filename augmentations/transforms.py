import albumentations as A
import cv2


def build_train_transforms(image_size=(1000, 1000)):
    height, width = image_size
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.2),
            A.Rotate(
                limit=90,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=25,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                value=0,
                mask_value=0,
                p=0.4,
            ),
            A.RandomResizedCrop(
                height=height,
                width=width,
                scale=(0.8, 1.0),
                ratio=(0.95, 1.05),
                interpolation=cv2.INTER_LINEAR,
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.10,
                        brightness_by_max=True,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=4,
                        sat_shift_limit=8,
                        val_shift_limit=8,
                        p=1.0,
                    ),
                    A.RGBShift(
                        r_shift_limit=8,
                        g_shift_limit=8,
                        b_shift_limit=8,
                        p=1.0,
                    ),
                ],
                p=0.45,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ],
                p=0.25,
            ),
        ]
    )


def build_eval_transforms(image_size=(1000, 1000)):
    del image_size
    return A.Compose([])
