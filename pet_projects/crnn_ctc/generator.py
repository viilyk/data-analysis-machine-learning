import os
import json
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import numpy as np
from trdg.generators import GeneratorFromRandom


# ========== Save helpers ==========
def save_metadata_and_labels(out_dir, labels, metadata=None):
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(labels))
    if metadata:
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


# ========== Core generation function ==========
def generate_level_chunk(args):
    out_dir, level_cfg, bg_type, count, offset = args
    images_dir = Path(out_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    metadata = {}
    idx = offset

    gen = GeneratorFromRandom(
        count=count,
        fonts=level_cfg.get("fonts"),
        allow_variable=True,
        size=level_cfg.get("size", 64),
        skewing_angle=level_cfg.get("skewing_angle", 0),
        random_skew=level_cfg.get("random_skew", False),
        blur=level_cfg.get("blur", 0),
        random_blur=level_cfg.get("random_blur", False),
        background_type=bg_type,
        distorsion_type=level_cfg.get("distorsion_type", 0),
        space_width=level_cfg.get("space_width", 1.0),
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for img_pil, label in gen:
            if img_pil is None:
                continue

            fname = f"{idx:08d}.jpg"
            out_path = images_dir / fname
            futures.append(executor.submit(img_pil.convert("RGB").save, out_path, quality=80, optimize=True))

            labels.append(f"{fname}\t{label}")
            metadata[fname] = {
                "label": label,
                "params": {
                    "size": level_cfg.get("size", 64),
                    "skewing_angle": level_cfg.get("skewing_angle", 0),
                    "random_skew": level_cfg.get("random_skew", False),
                    "blur": level_cfg.get("blur", 0),
                    "random_blur": level_cfg.get("random_blur", False),
                    "background_type": bg_type,
                    "distorsion_type": level_cfg.get("distorsion_type", 0),
                    "fonts_dir": level_cfg.get("fonts_dir"),
                },
            }
            idx += 1

    for f in futures:
        f.result()

    return labels, metadata


# ========== Основная функция генерации уровня ==========
def generate_level_parallel(out_dir, level_cfg):
    total_count = level_cfg.get("count", 1000)
    bg_probs = level_cfg.get("background_probs", {0: 1.0})

    bg_types, probs = zip(*bg_probs.items())
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()

    counts = (probs * total_count).astype(int).tolist()
    counts[0] += total_count - sum(counts)  # балансируем округления

    tasks, offset = [], 0
    for bg_type, c in zip(bg_types, counts):
        if c > 0:
            tasks.append((out_dir, level_cfg, bg_type, c, offset))
            offset += c

    with Pool(processes=min(len(tasks), cpu_count())) as pool:
        results = pool.map(generate_level_chunk, tasks)

    all_labels, all_metadata = [], {}
    for labels, metadata in results:
        all_labels.extend(labels)
        all_metadata.update(metadata)

    save_metadata_and_labels(out_dir, all_labels, all_metadata)
    return len(all_labels)


# ========== CLI and main execution ==========
def parse_args():
    p = argparse.ArgumentParser(description='TRDG dataset generator (curriculum A/B/C)')
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--counts', type=int, nargs=3, default=[50000, 30000, 20000], help='counts for A B C')
    p.add_argument('--size', type=int, default=64, help='image height')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    base = Path(args.out_dir)
    train = base / 'train'
    for lvl in ["A", "B", "C"]:
        (train / lvl / "images").mkdir(parents=True, exist_ok=True)

    font_dir = "fonts/"
    fonts_simple = [os.path.join(font_dir, 'simple', f) for f in os.listdir(font_dir + 'simple') if f.endswith((".ttf", ".otf"))]
    fonts_medium = [os.path.join(font_dir, 'medium', f) for f in os.listdir(font_dir + 'medium') if f.endswith((".ttf", ".otf"))]
    fonts_hard = [os.path.join(font_dir, 'hard', f) for f in os.listdir(font_dir + 'hard') if f.endswith((".ttf", ".otf"))]

    levels = {
        "A": {
            'count': args.counts[0], 'size': args.size, 'fonts': fonts_simple,
            'skewing_angle': 3, 'random_skew': True,
            'blur': 0, 'random_blur': False,
            'background_probs': {1: 0.7, 0: 0.2, 2: 0.1},
            'distorsion_type': 0, 'space_width': 1.0
        },
        "B": {
            'count': args.counts[1], 'size': args.size, 'fonts': fonts_medium,
            'skewing_angle': 10, 'random_skew': True,
            'blur': 1, 'random_blur': True,
            'background_probs': {0: 0.3, 1: 0.1, 2: 0.4, 3: 0.2},
            'distorsion_type': 3, 'space_width': 0.95
        },
        "C": {
            'count': args.counts[2], 'size': args.size, 'fonts': fonts_hard,
            'skewing_angle': 30, 'random_skew': True,
            'blur': 1.8, 'random_blur': True,
            'background_probs': {3: 0.45, 2: 0.35, 1: 0.1, 0: 0.1},
            'distorsion_type': 1, 'space_width': 0.9
        },
    }

    print("Generating train...")
    total_saved = 0
    for lvl, cfg in levels.items():
        print(f'Generating LEVEL {lvl}...')
        saved = generate_level_parallel(train / lvl, cfg)
        total_saved += saved
        print(f'TRAIN LEVEL {lvl} saved: {saved}')

    print(f'Total saved images in train: {total_saved}')


if __name__ == '__main__':
    main()
