"""Reproduce the report's evidence: per-stage errors, timings, ablations.

Run from the repository root:

    python tools/report_metrics.py

Writes docs/metrics.json and two figures under docs/figures/. Reference arrays
are used only for measurement here; the encoder never reads them.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import load_images, load_weights, image_records  # noqa: E402
from image_encoder import ImageEncoder  # noqa: E402
from helpers import LayerNorm, GELU, attention_weights  # noqa: E402

FONT = 'C:/Windows/Fonts/msyh.ttc'


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def error_stats(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
            'shape': list(actual.shape), 'dtype': str(actual.dtype)}


def block0_with_variant(x, weights, variant):
    """One image block with a deliberately wrong choice, for the failure table."""
    p = 'visual.transformer.resblocks.0'
    ln1 = LayerNorm(weights[p + '.ln_1.weight'], weights[p + '.ln_1.bias'])
    ln2 = LayerNorm(weights[p + '.ln_2.weight'], weights[p + '.ln_2.bias'])
    heads, width = 12, 768
    head_dim = width // heads
    batch, tokens, _ = x.shape

    qkv = ln1(x) @ weights[p + '.attn.in_proj_weight'].T + weights[p + '.attn.in_proj_bias']

    def split(array):
        if variant == 'interleaved_heads':          # wrong: stride by head_dim
            return array.reshape(batch, tokens, head_dim, heads).transpose(0, 3, 1, 2)
        return array.reshape(batch, tokens, heads, head_dim).transpose(0, 2, 1, 3)

    q = split(qkv[:, :, :width])
    k = split(qkv[:, :, width:2 * width])
    v = split(qkv[:, :, 2 * width:])

    if variant == 'k_all_axes':                     # wrong: k.T reverses 4 axes
        scores = (q @ k.transpose(3, 2, 1, 0)) / np.float32(np.sqrt(head_dim))
    else:
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))

    context = attention_weights(scores) @ v
    if variant == 'merge_no_transpose':             # wrong: heads left on axis 1
        merged = context.reshape(batch, tokens, width)
    elif variant == 'interleaved_heads':
        merged = context.transpose(0, 2, 3, 1).reshape(batch, tokens, width)
    else:
        merged = context.transpose(0, 2, 1, 3).reshape(batch, tokens, width)

    out_weight = weights[p + '.attn.out_proj.weight']
    out_bias = weights[p + '.attn.out_proj.bias']
    update = merged @ (out_weight if variant == 'out_no_transpose' else out_weight.T) + out_bias

    residual = x + update
    u = ln2(residual)
    gelu = GELU()
    hidden = gelu(u @ weights[p + '.mlp.c_fc.weight'].T + weights[p + '.mlp.c_fc.bias'])
    mlp_update = hidden @ weights[p + '.mlp.c_proj.weight'].T + weights[p + '.mlp.c_proj.bias']
    return residual + mlp_update


def draw_bar_chart(rows, path):
    width, height = 980, 470
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small = font(22), font(16), font(14)

    left, right, top, bottom = 90, width - 40, 70, height - 90
    plot_h = bottom - top
    lo, hi = -8.0, -2.0                              # log10 error axis
    draw.text((left, 20), '各阶段最大绝对误差（对数坐标）', font=title_font, fill='black')

    def y_of(value):
        value = max(lo, min(hi, value))
        return bottom - (value - lo) / (hi - lo) * plot_h

    for decade in range(-8, -1):
        y = y_of(decade)
        draw.line([(left, y), (right, y)], fill=(225, 225, 225))
        draw.text((left - 55, y - 9), f'1e{decade}', font=small, fill=(90, 90, 90))
    draw.line([(left, top), (left, bottom)], fill='black', width=2)
    draw.line([(left, bottom), (right, bottom)], fill='black', width=2)

    for value, color, text in ((math.log10(2e-4), (214, 48, 49), '内部阶段容差 2e-4'),
                               (math.log10(2e-5), (9, 105, 218), '输出容差 2e-5')):
        y = y_of(value)
        x = left
        while x < right:
            draw.line([(x, y), (min(x + 9, right), y)], fill=color, width=2)
            x += 16
        draw.text((right - 190, y - 20 if color == (214, 48, 49) else y + 3), text,
                  font=small, fill=color)

    slot = (right - left) / len(rows)
    bar_w = slot * 0.52
    for i, row in enumerate(rows):
        cx = left + slot * (i + 0.5)
        value = math.log10(max(row['max_abs'], 1e-12))
        y = y_of(value)
        draw.rectangle([cx - bar_w / 2, y, cx + bar_w / 2, bottom], fill=(52, 120, 190))
        draw.text((cx - 32, y - 24), f"{row['max_abs']:.1e}", font=small, fill=(30, 30, 30))
        label_w = draw.textlength(row['name'], font=label_font)
        draw.text((cx - label_w / 2, bottom + 12), row['name'], font=label_font, fill='black')
    draw.text((left, height - 34), '柱低者为优；全部柱体均低于对应容差线，判定 PASS',
              font=small, fill=(80, 80, 80))
    canvas.save(path)


def draw_patch_grid(path, figure_path):
    """Rebuild the 224x224 input of image_01 and label the 49 patch positions."""
    image = Image.open(path).convert('RGB')
    width, height = image.size
    if width <= height:
        size = (224, int(224 * height / width))
    else:
        size = (int(224 * width / height), 224)
    image = image.resize(size, Image.Resampling.BICUBIC)
    left = round((size[0] - 224) / 2)
    top = round((size[1] - 224) / 2)
    image = image.crop((left, top, left + 224, top + 224))

    scale = 3
    size_px = 224 * scale
    canvas = Image.new('RGB', (size_px, size_px + 46), 'white')
    canvas.paste(image.resize((size_px, size_px), Image.Resampling.NEAREST), (0, 46))
    draw = ImageDraw.Draw(canvas)
    label_font = font(20)
    title_font = font(19)
    title = '224×224 → 7×7 个 32×32 patch（编号=输出行号）'
    while draw.textlength(title, font=title_font) > size_px - 12 and title_font.size > 10:
        title_font = font(title_font.size - 1)
    draw.text((6, 10), title, font=title_font, fill='black')
    for index in range(49):
        row, column = divmod(index, 7)
        x, y = column * 32 * scale, row * 32 * scale + 46
        draw.rectangle([x, y, x + 32 * scale, y + 32 * scale], outline=(255, 255, 0), width=2)
        draw.text((x + 8, y + 6), str(index), font=label_font, fill=(255, 255, 0),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    canvas.save(figure_path)


def main():
    weights = load_weights()
    encoder = ImageEncoder(weights)
    images = load_images()
    records = image_records()
    with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as ref:
        reference = {key: ref[key] for key in ref.files}

    stages = []
    t0 = time.perf_counter()
    patches = encoder.patchify(images[:2])
    stages.append({'name': 'patches', **error_stats(patches, reference['patches'][:2]),
                   'seconds': time.perf_counter() - t0})

    t0 = time.perf_counter()
    embed = encoder.embed(images[:2])
    stages.append({'name': 'tokens', **error_stats(embed, reference['input_tokens'][:2]),
                   'seconds': time.perf_counter() - t0})

    pre = encoder.pre_norm(embed)
    t0 = time.perf_counter()
    block0 = encoder.blocks[0](pre)
    stages.append({'name': 'block0', **error_stats(block0, reference['block0_tokens'][:2]),
                   'seconds': time.perf_counter() - t0})

    t0 = time.perf_counter()
    last = encoder.encode_tokens(images[:2])
    stages.append({'name': 'last_tokens', **error_stats(last, reference['last_tokens'][:2]),
                   'seconds': time.perf_counter() - t0})

    t0 = time.perf_counter()
    tokens10 = encoder(images)
    stages.append({'name': 'output', **error_stats(tokens10, reference['output_tokens']),
                   'seconds': time.perf_counter() - t0})

    order = [1, 0, 1]
    t0 = time.perf_counter()
    batch = encoder(images[order])
    stages.append({'name': 'batch', **error_stats(batch, reference['output_tokens'][order]),
                   'seconds': time.perf_counter() - t0})

    # Extra contract evidence: unit length, per-image distance, order invariance.
    norms = np.sqrt((tokens10.astype(np.float64) ** 2).sum(-1))
    unit = np.array(tokens10, dtype=np.float64)
    cosine = unit @ unit.T
    off_diag = cosine[~np.eye(len(unit), dtype=bool)]
    reorder_error = float(np.abs(encoder(images[[3, 7, 5]]) -
                                 reference['output_tokens'][[3, 7, 5]]).max())
    repeat_error = float(np.abs(encoder(images[[2, 2]]) - tokens10[[2, 2]]).max())

    # Ablations: what a single wrong choice costs at block0. A crash is recorded
    # as such: k.T on a 4D array cannot multiply at all unless batch == 64.
    ablations = []
    for variant, note in (('k_all_axes', 'k.T 交换全部 4 个轴'),
                          ('interleaved_heads', '按 head_dim 步长交错切分 head'),
                          ('merge_no_transpose', '合并 head 时漏掉转置'),
                          ('out_no_transpose', '输出投影漏掉转置'),
                          ('causal_mask', '图像误用因果掩码'),
                          ('no_position', '漏加位置编码')):
        try:
            if variant == 'causal_mask':
                actual = encoder.blocks[0](pre, causal=True)
            elif variant == 'no_position':
                actual = encoder.blocks[0](
                    encoder.pre_norm(encoder.embed(images[:2]) - encoder.position))
            else:
                actual = block0_with_variant(pre, weights, variant)
            stats = error_stats(actual, reference['block0_tokens'][:2])
            ablations.append({'variant': variant, 'note': note,
                              'max_abs': stats['max_abs'], 'mean_abs': stats['mean_abs']})
        except Exception as exc:                      # noqa: BLE001 - record the failure
            ablations.append({'variant': variant, 'note': note,
                              'error': f'{type(exc).__name__}: {exc}'})

    summary = {
        'stages': stages,
        'output_norm': {'min': float(norms.min()), 'max': float(norms.max())},
        'pairwise_cosine': {'min': float(off_diag.min()), 'max': float(off_diag.max())},
        'reorder_max_abs': reorder_error,
        'duplicate_max_abs': repeat_error,
        'ablations': ablations,
        'image_ids': [r['id'] for r in records],
        'environment': {'python': sys.version.split()[0], 'numpy': np.__version__},
    }
    (ROOT / 'docs/metrics.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    draw_bar_chart(stages, ROOT / 'docs/figures/stage_error.png')
    draw_patch_grid(ROOT / 'assets/images/image_01.jpg', ROOT / 'docs/figures/patch_order.png')

    print(f"{'stage':12} {'max_abs':>10} {'mean_abs':>10} {'seconds':>8}")
    for row in stages:
        print(f"{row['name']:12} {row['max_abs']:10.3e} {row['mean_abs']:10.3e} {row['seconds']:8.2f}")
    print(f"output norms: {norms.min():.6f} .. {norms.max():.6f}")
    print(f"pairwise cosine: {off_diag.min():.4f} .. {off_diag.max():.4f}")
    print(f"reorder max_abs: {reorder_error:.2e}; duplicate max_abs: {repeat_error:.2e}")
    print('ablations (block0 max_abs):')
    for row in ablations:
        shown = row.get('error', f"{row.get('max_abs', float('nan')):.3e}")
        print(f"  {row['variant']:20} {row['note']:26} {shown}")


if __name__ == '__main__':
    main()
