#!/usr/bin/env python3
"""Batch inference and image-comparison utility for DiffUIR.

Examples
--------
# Run DiffUIR on every PNG/JPEG below INPUT, preserving relative paths.
python run_folder.py infer \
  --input-dir /path/to/noised/rgb_bright \
  --output-dir /path/to/diffuir_results \
  --checkpoint /path/to/model-300.pt

# After copying results to another machine, make input/restored panels.
python run_folder.py compare \
  --input-dir /path/to/noised/rgb_bright \
  --output-dir /path/to/diffuir_results \
  --comparison-dir /path/to/comparisons
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_paths(root: Path) -> Iterator[Path]:
    """Yield supported images recursively in a stable order."""
    yield from (path for path in sorted(root.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES)


def output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Return a PNG output path that mirrors the input folder structure."""
    return (output_dir / input_path.relative_to(input_dir)).with_suffix(".png")


def pad_to_multiple(tensor, multiple: int = 8):
    """Reflect-pad a BCHW tensor and return it with its unpadded height and width."""
    import torch.nn.functional as functional

    height, width = tensor.shape[-2:]
    pad_bottom = (-height) % multiple
    pad_right = (-width) % multiple
    if pad_bottom or pad_right:
        # Replication also works for very small images, unlike reflect padding.
        tensor = functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    return tensor, height, width


def load_diffuir(checkpoint: Path, sampling_timesteps: int):
    """Build the published Base model and load a DiffUIR Trainer checkpoint."""
    import torch
    from types import SimpleNamespace
    from src.visualization import ResidualDiffusion, Trainer, UnetRes, set_seed

    set_seed(10)
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=1,
        condition=True,
        objective="pred_res",
        test_res_or_noise="res",
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=256,
        timesteps=1000,
        delta_end=1.8e-3,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=0.0,
        objective="pred_res",
        loss_type="l1",
        condition=True,
        sum_scale=0.01,
        test_res_or_noise="res",
    )
    # Trainer is reused solely because the official checkpoint includes EMA state.
    trainer = Trainer(
        diffusion,
        dataset="unused-for-folder-inference",
        opts=SimpleNamespace(phase="test"),
        train_batch_size=1,
        num_samples=1,
        train_lr=2e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=str(checkpoint.parent),
        condition=True,
        save_and_sample_every=1000,
        num_unet=1,
    )
    data = torch.load(checkpoint, map_location=trainer.device)
    trainer.model = trainer.accelerator.unwrap_model(trainer.model)
    trainer.model.load_state_dict(data["model"])
    trainer.ema.load_state_dict(data["ema"])
    trainer.ema.ema_model.init()
    trainer.ema.to(trainer.device)
    trainer.ema.ema_model.eval()
    return trainer


def run_inference(args: argparse.Namespace) -> None:
    from PIL import Image
    import torch
    from torchvision.transforms.functional import to_tensor

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    trainer = load_diffuir(checkpoint, args.sampling_timesteps)
    inputs = list(image_paths(args.input_dir))
    if not inputs:
        raise FileNotFoundError(f"No supported images below {args.input_dir}")

    for index, input_path in enumerate(inputs, start=1):
        destination = output_path(input_path, args.input_dir, args.output_dir)
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(inputs)}] skipping {destination}")
            continue
        with Image.open(input_path) as image:
            image = image.convert("RGB")
            tensor = to_tensor(image).unsqueeze(0).to(trainer.device)
        tensor, original_height, original_width = pad_to_multiple(tensor)
        with torch.no_grad():
            samples = list(trainer.ema.ema_model.sample(tensor, batch_size=1, last=True, task=None))
            restored = samples[-1][..., :original_height, :original_width]
        destination.parent.mkdir(parents=True, exist_ok=True)
        # save_image clamps to [0, 1], matching the repository's visual.py behaviour.
        from torchvision.utils import save_image

        save_image(restored, destination)
        print(f"[{index}/{len(inputs)}] saved {destination}")


def add_label(image, label: str):
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 24), fill="black")
    draw.text((6, 5), label, fill="white")
    return canvas


def make_comparisons(args: argparse.Namespace) -> None:
    from PIL import Image

    inputs = list(image_paths(args.input_dir))
    if not inputs:
        raise FileNotFoundError(f"No supported images below {args.input_dir}")
    made = 0
    for input_path in inputs:
        restored_path = output_path(input_path, args.input_dir, args.output_dir)
        if not restored_path.is_file():
            print(f"missing restored file: {restored_path}")
            continue
        paths_and_labels = [(input_path, "Input"), (restored_path, "DiffUIR")]
        if args.reference_dir:
            reference_path = output_path(input_path, args.input_dir, args.reference_dir)
            if reference_path.is_file():
                paths_and_labels.append((reference_path, "Reference"))
            else:
                print(f"missing reference file: {reference_path}")
        panels = []
        for path, label in paths_and_labels:
            with Image.open(path) as image:
                panels.append(add_label(image.convert("RGB"), label))
        height = max(panel.height for panel in panels)
        resized = [panel.resize((round(panel.width * height / panel.height), height)) for panel in panels]
        comparison = Image.new("RGB", (sum(panel.width for panel in resized), height), "black")
        cursor = 0
        for panel in resized:
            comparison.paste(panel, (cursor, 0))
            cursor += panel.width
        destination = output_path(input_path, args.input_dir, args.comparison_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(destination)
        made += 1
    print(f"Created {made} comparison image(s) in {args.comparison_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch DiffUIR inference and comparisons.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-dir", type=Path, required=True, help="Directory of RGB PNG/JPEG/TIFF inputs.")
    common.add_argument("--output-dir", type=Path, required=True, help="DiffUIR output directory.")

    infer = subparsers.add_parser("infer", parents=[common], help="Run the DiffUIR Base checkpoint.")
    infer.add_argument("--checkpoint", type=Path, required=True, help="Path to model-300.pt.")
    infer.add_argument("--sampling-timesteps", type=int, default=3, help="Use 3 for model-300.pt.")
    infer.add_argument("--overwrite", action="store_true", help="Regenerate existing output PNGs.")

    compare = subparsers.add_parser("compare", parents=[common], help="Create labelled side-by-side panels.")
    compare.add_argument("--comparison-dir", type=Path, required=True, help="Where comparison PNGs are saved.")
    compare.add_argument("--reference-dir", type=Path, help="Optional third panel (ground truth/reference), mirroring input names.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "infer":
        run_inference(arguments)
    else:
        make_comparisons(arguments)
