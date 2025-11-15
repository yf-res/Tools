#!/usr/bin/env python3
"""
Multi-frame super-resolution from a stack of aligned images.

Usage examples:

# Classical (OpenCV-based)
python multi_superres.py classical \
    --input_dir path/to/images \
    --output path/to/output.png \
    --scale 2 \
    --merge_method mean \
    --no_registration

# Deep-learning (EDVR-like, template)
python multi_superres.py deep \
    --input_dir path/to/images \
    --output path/to/output.png \
    --scale 4 \
    --model_path path/to/edvr_model.pth \
    --device cuda

"""

import os
import glob
import argparse

import numpy as np
import cv2


# -------------------------------
# Common utilities
# -------------------------------

def load_images_from_dir(input_dir, max_images=None):
    """Load all images from a directory as float32 BGR arrays in [0, 1]."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files = sorted(files)

    if not files:
        raise RuntimeError(f"No images found in directory: {input_dir}")

    if max_images is not None:
        files = files[:max_images]

    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to read {f}, skipping.")
            continue
        imgs.append(img.astype(np.float32) / 255.0)

    if not imgs:
        raise RuntimeError("No valid images loaded.")

    print(f"Loaded {len(imgs)} images from {input_dir}")
    return imgs


def save_image(path, img_float):
    """Save an image float32 [0,1] or [0,255] as uint8 PNG/JPG."""
    if img_float.dtype != np.uint8:
        img = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img_float
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Saved: {path}")


# -------------------------------
# Approach 1: Classical Multi-Frame SR
# -------------------------------

def register_images_ecc(ref_hr, img_hr, motion_type=cv2.MOTION_AFFINE, number_of_iterations=100, termination_eps=1e-6):
    """
    Refine alignment of img_hr to ref_hr using ECC.
    Both images must be single-channel or 3-channel float32 in [0,1].
    Returns the aligned image.
    """
    if ref_hr.ndim == 3:
        ref_gray = cv2.cvtColor(ref_hr, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_hr

    if img_hr.ndim == 3:
        img_gray = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_hr

    warp_matrix = np.eye(2, 3, dtype=np.float32)  # for MOTION_AFFINE
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_gray, img_gray, warp_matrix, motion_type, criteria
        )
        aligned = cv2.warpAffine(
            img_hr,
            warp_matrix,
            (ref_hr.shape[1], ref_hr.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
        return aligned
    except cv2.error as e:
        print("ECC failed, returning unaligned image. Error:", e)
        return img_hr


def merge_images(registered_imgs, method="mean"):
    """Merge multiple registered high-resolution images."""
    stack = np.stack(registered_imgs, axis=0)  # (N, H, W, C)
    if method == "mean":
        merged = np.mean(stack, axis=0)
    elif method == "median":
        merged = np.median(stack, axis=0)
    else:
        raise ValueError("merge_method must be 'mean' or 'median'")
    return merged


def unsharp_mask(img, sigma=1.0, strength=1.0):
    """Simple unsharp masking for sharpening."""
    blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    # sharpened = img * (1 + strength) - blur * strength
    sharpened = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
    return np.clip(sharpened, 0.0, 1.0)


def classical_super_resolution(
    imgs,
    scale=2,
    do_registration=True,
    merge_method="mean",
    sharpen=True,
    sharpen_sigma=1.0,
    sharpen_strength=1.0,
):
    """
    Classical multi-frame super-resolution pipeline:
    1. Upscale each image.
    2. (Optional) ECC registration.
    3. Merge (mean or median).
    4. (Optional) Sharpen.
    """
    ref = imgs[0]
    h, w = ref.shape[:2]
    print(f"Base size: {w}x{h}, upscaling by factor {scale}")
    hr_size = (w * scale, h * scale)

    # Upscale all images
    upscaled = []
    for i, im in enumerate(imgs):
        up = cv2.resize(im, hr_size, interpolation=cv2.INTER_CUBIC)
        upscaled.append(up)
        print(f"Upscaled image {i+1}/{len(imgs)}")

    # Registration
    if do_registration:
        print("Refining alignment with ECC...")
        ref_hr = upscaled[0]
        registered = [ref_hr]
        for i in range(1, len(upscaled)):
            aligned = register_images_ecc(ref_hr, upscaled[i])
            registered.append(aligned)
    else:
        print("Skipping registration. Assuming images are already well aligned at HR scale.")
        registered = upscaled

    # Merge
    print(f"Merging {len(registered)} images with method '{merge_method}'...")
    merged = merge_images(registered, method=merge_method)

    # Sharpen
    if sharpen:
        print("Applying unsharp mask sharpening...")
        merged = unsharp_mask(merged, sigma=sharpen_sigma, strength=sharpen_strength)

    return merged


# -------------------------------
# Approach 2: Deep-Learning Multi-Frame SR (EDVR-style template)
# -------------------------------

def deep_super_resolution_edvr(
    imgs,
    scale=4,
    model_path="edvr_model.pth",
    device="cuda",
    num_frames=None,
):
    """
    Deep-learning multi-frame SR using an EDVR-like model.

    IMPORTANT:
    - This function assumes you have an EDVR implementation available.
    - You must adjust the imports below to match your EDVR code.
    - The checkpoint at `model_path` must match the EDVR architecture you load.

    Typical setup:
    - git clone https://github.com/xinntao/EDVR.git
    - Add the repo to PYTHONPATH, or install it as a package.
    """
    try:
        import torch
        from torch import nn

        # ---- TODO: adjust this import to your EDVR implementation ----
        # Example if you're using the official EDVR repo:
        # from edvr.archs.edvr_arch import EDVR
        #
        # For this template, we'll define a dummy EDVR class that you should replace.

        class DummyEDVR(nn.Module):
            """
            Placeholder EDVR-like network.
            Replace this with `from edvr.archs.edvr_arch import EDVR`
            and construct EDVR with correct arguments.
            """

            def __init__(self):
                super().__init__()
                # This is NOT a real EDVR, just a placeholder 1x1 conv
                # so the script is syntactically complete.
                self.conv = nn.Conv2d(3, 3 * (scale ** 2), kernel_size=3, padding=1)
                self.pixel_shuffle = nn.PixelShuffle(scale)

            def forward(self, x):
                # x: (B, T, C, H, W)
                # We'll just use the center frame for this dummy.
                # A real EDVR uses all frames jointly.
                center = x[:, x.shape[1] // 2]  # (B, C, H, W)
                y = self.conv(center)
                y = self.pixel_shuffle(y)
                return y

        # Replace DummyEDVR() with your real EDVR(...) constructor:
        # model = EDVR(...)

        print("Building EDVR model (template/dummy)...")
        model = DummyEDVR()

        if os.path.isfile(model_path):
            print(f"Loading model weights from: {model_path}")
            state = torch.load(model_path, map_location=device)
            # Depending on how you saved the checkpoint, you may need:
            #   model.load_state_dict(state, strict=False)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        else:
            print(f"Warning: model_path '{model_path}' not found. Using random weights (for demo only).")

        model.to(device)
        model.eval()

        # Preprocess images -> tensor (1, T, C, H, W), normalized to [0,1]
        if num_frames is not None:
            imgs_use = imgs[:num_frames]
        else:
            imgs_use = imgs

        print(f"Using {len(imgs_use)} frames for deep SR.")

        h, w = imgs_use[0].shape[:2]
        # Ensure all images same size
        for i, im in enumerate(imgs_use):
            if im.shape[:2] != (h, w):
                raise RuntimeError("All images must have same size for deep SR.")

        # Stack images: (T, H, W, C) -> (1, T, C, H, W)
        frames = np.stack(imgs_use, axis=0)  # (T, H, W, C)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
        frames_tensor = torch.from_numpy(frames).float().unsqueeze(0)  # (1, T, C, H, W)

        frames_tensor = frames_tensor.to(device)

        with torch.no_grad():
            sr = model(frames_tensor)  # expect (1, C, H*scale, W*scale)

        sr = sr.squeeze(0).cpu().numpy()  # (C, H', W')
        sr = np.transpose(sr, (1, 2, 0))  # (H', W', C)
        sr = np.clip(sr, 0.0, 1.0)  # assuming model outputs [0,1]; adjust if needed

        return sr

    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for the deep-learning approach. "
            "Install it with `pip install torch` or `conda install pytorch`.\n"
            f"Original error: {e}"
        )


# -------------------------------
# CLI
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-frame super-resolution from aligned images."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Classical
    p_classical = subparsers.add_parser("classical", help="Classical OpenCV-based multi-frame SR")
    p_classical.add_argument("--input_dir", required=True, help="Directory with input images")
    p_classical.add_argument("--output", required=True, help="Output image path")
    p_classical.add_argument("--scale", type=int, default=2, help="Upscale factor (default: 2)")
    p_classical.add_argument(
        "--merge_method",
        choices=["mean", "median"],
        default="mean",
        help="Merge method for multi-frame fusion (default: mean)",
    )
    p_classical.add_argument(
        "--no_registration",
        action="store_true",
        help="Disable ECC registration (assume perfectly aligned images)",
    )
    p_classical.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply unsharp mask sharpening at the end",
    )
    p_classical.add_argument("--sharpen_sigma", type=float, default=1.0, help="Sharpening Gaussian sigma")
    p_classical.add_argument("--sharpen_strength", type=float, default=1.0, help="Sharpening strength")
    p_classical.add_argument("--max_images", type=int, default=None, help="Use at most N images")

    # Deep
    p_deep = subparsers.add_parser("deep", help="Deep-learning (EDVR-style) multi-frame SR")
    p_deep.add_argument("--input_dir", required=True, help="Directory with input images")
    p_deep.add_argument("--output", required=True, help="Output image path")
    p_deep.add_argument("--scale", type=int, default=4, help="Upscale factor (used in dummy model)")
    p_deep.add_argument("--model_path", type=str, default="edvr_model.pth", help="Path to EDVR model checkpoint")
    p_deep.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    p_deep.add_argument("--max_images", type=int, default=None, help="Use at most N images")
    p_deep.add_argument("--num_frames", type=int, default=None, help="Limit number of frames fed to the model")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "classical":
        imgs = load_images_from_dir(args.input_dir, max_images=args.max_images)
        sr = classical_super_resolution(
            imgs,
            scale=args.scale,
            do_registration=not args.no_registration,
            merge_method=args.merge_method,
            sharpen=args.sharpen,
            sharpen_sigma=args.sharpen_sigma,
            sharpen_strength=args.sharpen_strength,
        )
        save_image(args.output, sr)

    elif args.mode == "deep":
        imgs = load_images_from_dir(args.input_dir, max_images=args.max_images)
        sr = deep_super_resolution_edvr(
            imgs,
            scale=args.scale,
            model_path=args.model_path,
            device=args.device,
            num_frames=args.num_frames,
        )
        save_image(args.output, sr)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
