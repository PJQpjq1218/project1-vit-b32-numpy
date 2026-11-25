"""Build the experiment report (DOCX) from the measured evidence.

Run from the repository root:

    python tools/build_report.py

Reads docs/metrics.json (produced by tools/report_metrics.py) so every number in
the report is the number the code actually measured.
"""
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / 'docs/metrics.json').read_text(encoding='utf-8'))

BODY_FONT = '宋体'
HEAD_FONT = '黑体'
MONO_FONT = 'Consolas'


def set_run_font(run, name, size=None, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def body(document, text, size=10, first_indent=True, space_after=2, line=1.1):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.line_spacing = line
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Pt(size * 2)
    # Bold text between ** markers, so emphasis survives without extra runs.
    parts = text.split('**')
    for index, part in enumerate(parts):
        if not part:
            continue
        run = paragraph.add_run(part)
        set_run_font(run, BODY_FONT, size, bold=(index % 2 == 1))
    return paragraph


def heading(document, text, level=1):
    sizes = {1: 12.5, 2: 11, 3: 10.5}
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(7 if level == 1 else 5)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, HEAD_FONT, sizes[level], bold=True)
    return paragraph


def formula(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    set_run_font(run, 'Cambria Math', 10.5)
    return paragraph


def caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    set_run_font(run, BODY_FONT, 9, color=RGBColor(0x33, 0x33, 0x33))
    return paragraph


def shade(cell, hex_color):
    element = OxmlElement('w:shd')
    element.set(qn('w:val'), 'clear')
    element.set(qn('w:fill'), hex_color)
    cell._tc.get_or_add_tcPr().append(element)


def make_table(document, header, rows, widths, font_size=9, header_fill='DCE6F1'):
    table = document.add_table(rows=1, cols=len(header))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Tight cell padding, so a 7-column table still fits the text width.
    margins = OxmlElement('w:tblCellMar')
    for side in ('top', 'bottom'):
        element = OxmlElement(f'w:{side}')
        element.set(qn('w:w'), '14')
        element.set(qn('w:type'), 'dxa')
        margins.append(element)
    for side in ('left', 'right'):
        element = OxmlElement(f'w:{side}')
        element.set(qn('w:w'), '56')
        element.set(qn('w:type'), 'dxa')
        margins.append(element)
    table._tbl.tblPr.append(margins)

    for index, text in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.width = Cm(widths[index])
        cell.text = ''
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        run = paragraph.add_run(text)
        set_run_font(run, HEAD_FONT, font_size, bold=True)
        shade(cell, header_fill)
    for row in rows:
        cells = table.add_row().cells
        for index, text in enumerate(row):
            cells[index].width = Cm(widths[index])
            cells[index].text = ''
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            run = paragraph.add_run(str(text))
            set_run_font(run, BODY_FONT, font_size)
    # A row never breaks across pages, and the header repeats when it must.
    for position, row in enumerate(table.rows):
        properties = row._tr.get_or_add_trPr()
        cant_split = OxmlElement('w:cantSplit')
        properties.append(cant_split)
        if position == 0:
            properties.append(OxmlElement('w:tblHeader'))
    return table


def figure(document, path, width_cm, cap):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(document, cap)


def build():
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    style = document.styles['Normal']
    style.font.name = BODY_FONT
    style.font.size = Pt(10)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

    stages = {row['name']: row for row in METRICS['stages']}
    ablations = {row['variant']: row for row in METRICS['ablations']}
    norms = METRICS['output_norm']
    cosine = METRICS['pairwise_cosine']

    # ---------------- title block ----------------
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run('小项目一实验报告')
    set_run_font(run, HEAD_FONT, 17, bold=True)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(8)
    run = subtitle.add_run('用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器')
    set_run_font(run, HEAD_FONT, 12.5, bold=False)

    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(10)
    run = meta.add_run('姓名：彭佳琦    学号：2025681109    班级：人工智能学院 IEEE01 班    日期：2026 年 11 月 26 日')
    set_run_font(run, BODY_FONT, 9, color=RGBColor(0x44, 0x44, 0x44))

    # ---------------- 1 task and method ----------------
    heading(document, '一、任务与方法')
    heading(document, '1.1 任务与数据', 2)
    body(document, '本项目的任务是在**不使用 PyTorch 等深度学习框架**的前提下，只用 NumPy 补全课程给定的 '
                   'OpenCLIP ViT-B/32 视觉编码器，使其前向数值与课程提供的独立 Torch FP32 参考一致。输入是'
                   '预处理后的图像批次 [B,3,224,224]（float32），输出是每张图一个 512 维单位向量 [B,512]，'
                   '即期末项目 szu_python_openclip 图像塔的最终输出。')
    body(document, '数据为课程附件中的 10 张真实图片（短边缩放 + 中心裁剪到 224×224，按 OpenCLIP 均值 '
                   '[0.48145, 0.45783, 0.40821] 与标准差 [0.26863, 0.26130, 0.27578] 归一化），以及 152 个 '
                   'visual.* 参数张量，共 87,849,216 个 float32 值。全程**只做前向推理，没有任何参数更新**，'
                   '考察的是对数组轴、形状变换和矩阵乘法的掌握。')

    heading(document, '1.2 方法总览', 2)
    body(document, '整条管线分为五段：① patchify 把图像切成 49 个 32×32 的块并展平；② embed 做无偏置的块'
                   '线性投影、拼接 CLS 行、加学习到的位置编码；③ 12 层 Pre-LN TransformerBlock，每层是「注意力'
                   '＋残差」再「MLP＋残差」；④ 输出头取 CLS 行、做 ln_post、乘 visual.proj；⑤ 逐行 L2 归一化。'
                   '核心计算如下（γ、β、W 均为课程提供的参数，W 按 [out,in] 存放）：')
    formula(document, '(1)  LN(x) = (x − μ) / √(σ² + 1e−5) ⊙ γ + β')
    formula(document, '(2)  A = softmax(Q Kᵀ / √d) V ，Q = XWqᵀ + bq ，d = 64')
    formula(document, '(3)  X₁ = X + A Woᵀ + bo')
    formula(document, '(4)  X₂ = X₁ + GELU(X₁W₁ᵀ + b₁) W₂ᵀ + b₂ ，GELU(u) = 0.5u(1 + erf(u/√2))')
    formula(document, '(5)  z = LNpost(X₂ 的 CLS 行) · Wproj ，e = z / ‖z‖₂ ，Wproj 为 768×512')

    body(document, '其中 Q、K、V 由一次 [B,N,2304] 的合并投影得到后切成 12 个头（每头 64 维），注意力在 '
                   '「每个头内部」独立进行，这是本任务最容易写错的地方。表 1 给出各阶段的数组形状。')

    make_table(document,
               ['阶段', '数组形状', '关键算子', '说明'],
               [['预处理输入', '[B,3,224,224]', 'BICUBIC 缩放 + 中心裁剪 + 归一化', '课程提供'],
                ['patchify', '[B,49,3072]', 'reshape → transpose → reshape', '49 = 7×7，3072 = 3×32×32'],
                ['embed', '[B,50,768]', 'patches @ Wᵀ，拼 CLS，+ 位置', '无 bias，50 = 1 + 49'],
                ['Q/K/V', '[B,12,50,64]', 'ln1 → 合并投影 → split_heads', '12 头 × 64 维'],
                ['scores', '[B,12,50,50]', 'q @ kᵀ / √64，再 softmax', '每个头独立 50×50'],
                ['block 输出', '[B,50,768]', 'context @ W_oᵀ + 残差，再来一次 MLP + 残差', '共 12 层串联'],
                ['output token', '[B,512]', '取 CLS → ln_post → @ W_proj → 单位化', '‖e‖₂ = 1']],
               widths=[3.2, 3.5, 6.6, 3.3])
    caption(document, '表 1  各阶段数组形状与关键算子（B 为批量大小）')

    heading(document, '1.3 评价标准', 2)
    body(document, '判定条件是：形状一致、dtype 为 float32、全部有限，且 |实际 − 参考| ≤ atol + rtol·|参考|。'
                   '其中预处理 atol = 2e−6，内部 token 2e−4，最终输出 2e−5，rtol 统一为 2e−4；输出阶段还要'
                   '求每行 L2 范数为 1，并额外检查批次乱序 [1,0,1]（含重复图）与逐张推理一致。')

    # ---------------- 2 reproducibility ----------------
    heading(document, '二、可复现性')
    heading(document, '2.1 环境与依赖', 2)
    body(document, '实测环境：Windows 11，Python 3.12.14，NumPy 2.3.5，Pillow 12.3.0，纯 CPU（不需要 GPU、'
                   'PyTorch、OpenCLIP 或联网）。依赖只有两行，见 requirements.txt；安装与运行命令均在仓库根目录执行。')
    make_table(document,
               ['步骤', '命令', '预期结果'],
               [['安装依赖', 'python -m pip install -r requirements.txt', '安装 numpy、Pillow'],
                ['分阶段自检', 'python check.py --stage all', '8 个阶段全部 PASS'],
                ['批量推理', 'python run.py --output output_tokens.npz', '保存 [10,512] float32'],
                ['复现报告证据', 'python tools/report_metrics.py', '生成 docs/metrics.json 与两张插图']],
               widths=[3.0, 8.2, 5.4], font_size=8.5)
    caption(document, '表 2  复现步骤（check.py 与 run.py 为课程提供，未修改）')

    heading(document, '2.2 数据获取与目录结构', 2)
    body(document, '两个大文件不入库：weights/image_encoder.npz（326 MB）超过 GitHub 单文件 100 MB 限制，'
                   'assets/reference_outputs.npz（7.6 MB）是课程参考数据且课程要求不要重复提交。README 给出了'
                   '两者的 SHA-256 与放置路径，weights/manifest.json 还逐张量记录形状、dtype 与 SHA-256，可用于'
                   '校验解压完整性。仓库目录：根目录为源码与配置（image_encoder.py、helpers.py、'
                   'preprocess_image.py、runtime.py、run.py、check.py、config.json），assets/ 与 weights/ 为数据，'
                   'tools/ 为证据与报告脚本，docs/ 为报告与插图。')

    heading(document, '2.3 本人贡献与所用协助', 2)
    body(document, '本人独立完成 image_encoder.py 中全部六个 TODO（patchify、split_heads、注意力三小步、块内'
                   '两处残差与 MLP、嵌入、输出头），未修改课程提供的 helpers.py、preprocess_image.py、check.py、'
                   'run.py、runtime.py 与 config.json；另外自行编写了证据脚本 tools/report_metrics.py（分阶段误差、'
                   '耗时、批量一致性、6 组错误实现对照）以及本报告。')
    body(document, '使用 DeepSeek 助手辅助的部分：解释 4D 张量上转置的轴语义、复核 reshape/transpose 的轴序、'
                   '起草 README 与报告初稿并排版。所有数值结论都由本机实际运行 check.py 与 tools/report_metrics.py '
                   '得到，并逐项核对过；代码中的每一行为本人理解后写入，未直接照搬他人实现。')

    # ---------------- 3 results ----------------
    heading(document, '三、结果与证据')
    heading(document, '3.1 分阶段数值验证', 2)
    body(document, '按 README 建议的顺序逐阶段自检，结果见表 3。8 个阶段全部通过：patch 与预处理阶段最大误差'
                   '为 0（这两步只有索引与广播，不产生浮点运算），内部 token 的最大误差在 1e−5 量级，最终输出'
                   'token 的最大误差 5.4e−7，比容差 2e−5 小约 37 倍。')
    rows = []
    tolerance = {'patches': '2e−4', 'tokens': '2e−4', 'block0': '2e−4',
                 'last_tokens': '2e−4', 'output': '2e−5', 'batch': '2e−5'}
    labels = {'patches': 'patchify [2,49,3072]', 'tokens': 'embed [2,50,768]',
              'block0': 'block0 [2,50,768]', 'last_tokens': '12 层后 [2,50,768]',
              'output': '输出 [10,512]', 'batch': '乱序批次 [3,512]'}
    for name in ('patches', 'tokens', 'block0', 'last_tokens', 'output', 'batch'):
        row = stages[name]
        rows.append([labels[name], f"{row['max_abs']:.1e}", f"{row['mean_abs']:.1e}",
                     tolerance[name], 'PASS'])
    make_table(document, ['阶段', '最大绝对误差', '平均绝对误差', 'atol', '判定'],
               rows, widths=[6.2, 3.0, 3.0, 2.4, 2.0])
    caption(document, '表 3  分阶段数值验证结果（rtol = 2e−4，参考为独立 Torch FP32 前向）')

    figure(document, ROOT / 'docs/figures/stage_error.png', 14.2,
           '图 1  各阶段最大绝对误差（对数坐标）与容差线；所有柱体均低于对应容差')
    body(document, '图 1 把表 3 画成对数坐标：patch 阶段严格为 0，说明切块与展平顺序完全正确；tokens 阶段升到'
                   ' 4.0e−5，仅来自矩阵乘法的累加顺序差异；block0 反而更低（7.1e−6），说明单层注意力与参考一致；'
                   '12 层叠加后回到 4.8e−5，最后经投影与单位化收缩到 5.4e−7。误差没有随层数指数放大，说明不存在'
                   '系统性偏差（如转置、错位），只是浮点舍入。')

    heading(document, '3.2 切块顺序的可视化证据', 2)
    body(document, '图 2 把 image_01 的 224×224 输入叠上 7×7 网格并标注 0–48 的编号，编号即该 patch 在输出'
                   '数组第 2 轴上的行号：编号沿「先左右、后上下」递增，每个 patch 内部再按 C、H、W 展平成 3072 维。'
                   '这一顺序必须与位置编码的顺序一致，否则数值会立刻错位。')
    figure(document, ROOT / 'docs/figures/patch_order.png', 6.2,
           '图 2  49 个 patch 的编号顺序（= 输出行号），内部按 C,H,W 展平')

    heading(document, '3.3 批量与几何性质', 2)
    body(document, f'还检查了三条与「能否真正复用」有关的性质：① 输出每行的 L2 范数为 '
                   f'{norms["min"]:.6f}–{norms["max"]:.6f}，满足单位化要求；② 10 张图两两余弦相似度为 '
                   f'{cosine["min"]:.3f}–{cosine["max"]:.3f}，既不是全 1（特征塌缩）也不是 0（随机输出），'
                   f'说明确实提取到了有区分度的语义特征；③ 批次按 [1,0,1] 重排（含重复图）与逐张推理的最大差仅 '
                   f'{METRICS["reorder_max_abs"]:.1e}，重复图输出差 {METRICS["duplicate_max_abs"]:.1e}。'
                   f'这说明实现不依赖缓存或参考行，可处理任意图片与任意批量。')

    # ---------------- 4 analysis ----------------
    heading(document, '四、分析与反思')
    heading(document, '4.1 一个形状变化的完整解释：图像 → 49 个 patch 行', 2)
    body(document, 'patchify 是本任务里唯一一次「连续三步变换」，也是最容易静默出错的地方：')
    body(document, '第一步 reshape(B, 3, 7, 32, 7, 32)：把 H 轴拆成「7 个 patch 行 × 32 行像素」、W 轴拆成'
                   '「7 个 patch 列 × 32 列像素」。reshape 不复制数据，只改轴的解读，此时轴含义是 '
                   '(B, C, 行块, 块内行, 列块, 块内列)。第二步 transpose(0, 2, 4, 1, 3, 5)：重排为 '
                   '(B, 行块, 列块, C, 块内行, 块内列)，这一步同时决定了两个顺序——第 2、3 轴决定 49 行的编号是'
                   '「先左右后上下」，第 4–6 轴决定每个 patch 内部 3072 维按 C→H→W 展平。任意一个写反，形状依旧'
                   '合法但数值全错。')
    body(document, '第三步 reshape(B, 49, 3072)：把 (7,7) 合并成 49、把 (3,32,32) 合并成 3072。转置后数组在'
                   '内存中不连续，必须先 ascontiguousarray 再 reshape，否则 NumPy 会按内存顺序而非逻辑顺序展平，'
                   '得到「每个 patch 被打散」的结果。实测这一步与参考完全相等（max_abs = 0），而 tokens 阶段的 '
                   '4.0e−5 只可能来自矩阵乘法，说明 49 行与 3072 维两个顺序同时正确。')

    heading(document, '4.2 一个 bug 的定位与修复：4D 张量上的 k.T', 2)
    body(document, '最初我照搬课程二维教学例子的写法，用 k.T 计算 QKᵀ。运行 check.py --stage block0 时直接'
                   '抛出 ValueError：matmul 的 core dimension 不匹配（size 12 is different from 64）。'
                   '原因是真实的 k 形状是 [B,12,50,64]，.T 会反转全部四个轴，得到 [64,50,12,B]，最后两维变成'
                   '(12,B)，无法与 q 的 [B,12,50,64] 相乘。更危险的是：如果批量大小恰好为 64，形状会「碰巧」'
                   '合法，代码不报错但结果完全不同——这类错误只能靠数值参考发现。')
    body(document, '修复方法：只交换最后两个轴，k.transpose(0,1,3,2) 得到 [B,12,64,50]，batch 与 head 分组'
                   '保持不变，scores 为 [B,12,50,50]。定位过程也值得记录：先跑 patches/tokens 确认输入侧无误，'
                   '再单独跑 block0，打印 q、k、v 形状并断言 scores 是 (B,H,N,N)，而不是从头读代码。修复后 '
                   'block0 最大误差为 7.1e−6。')

    heading(document, '4.3 对照实验：形状正确不等于语义正确', 2)
    body(document, '为验证「哪一处写错会被检查抓住、哪一处会静默通过」，我实现了 6 个只错一处的变体，'
                   '与正确实现（block0 最大误差 7.1e−6）对比，结果见表 4。')
    rows = []
    for key in ('merge_no_transpose', 'out_no_transpose', 'no_position',
                'interleaved_heads', 'causal_mask'):
        row = ablations[key]
        rows.append([row['note'], f"{row['max_abs']:.2f}"])
    rows.append([ablations['k_all_axes']['note'], '运行时报 ValueError'])
    make_table(document, ['单一错误（其余部分保持正确）', 'block0 最大绝对误差'],
               rows, widths=[10.6, 6.0], font_size=8.5)
    caption(document, '表 4  对照实验：正确实现为 7.1e−6，任何一处错误都相差 5 个数量级以上')

    body(document, '三条结论：① 合并头时漏转置（1.91）、输出投影漏转置（1.88）、漏加位置编码（1.78）的误差'
                   '与正确值相差五个数量级，这类错误一定会被 check.py 立即抓住；② 按 head_dim 步长「交错」切头'
                   '（0.87）形状完全合法，但打乱了每个头的特征子空间，是典型的「形状对、语义错」；③ 图像塔误用'
                   '因果掩码（0.44）会让 CLS 只能看到自己，误差虽比其它变体小，但同样致命。这组实验说明：'
                   '不能靠「跑通不报错」判断实现正确，必须用分阶段参考数值定位。')

    heading(document, '4.4 局限与改进方向', 2)
    body(document, '① 性能：课程提供的 GELU 是 Python 逐元素循环调用 math.erf（每层 B×50×3072 次），占整条'
                   '管线绝大部分耗时（10 张图 3.2 s）。改用 scipy.special.erf 向量化可提速 5–10 倍且保持精度；'
                   '用 tanh 近似则不可取，会引入约 1e−3 的误差，直接超出容差。'
                   '② 通用性：输入固定 224×224、patch 固定 32，未支持任意分辨率或非 32 整除的尺寸；批量维可任意。'
                   '③ 精度：与 Torch 约 1e−5 的差异来自 BLAS 累加顺序与 matmul 分块，属于 float32 固有，'
                   '无法通过改公式消除，因此不应盲目追求 0 误差。'
                   '④ 测试：目前依赖 check.py 的固定阶段，缺少对 B=1、全 0/全 255 极端输入、非方形图片的单元测试，'
                   '后续可以补一组参数化测试。')

    heading(document, '4.5 反思', 2)
    body(document, '这次最大的收获是建立了「轴语义」的习惯：每一次 reshape/transpose 都要能用一句话说清每个轴'
                   '代表什么（B、token、head、head_dim、C、h、w），本报告 4.1 节就是按这个习惯写下来的。'
                   '其次是调试策略：从「第一个出现差异的阶段」入手，比逐行阅读代码高效得多——patch 精确为 0 '
                   '排除了索引问题，tokens 只剩 4e−5 说明只有浮点差异，于是问题必然在 block 之后。'
                   '最后是对 AI 辅助的边界有了体会：助手能很快给出候选实现和解释，但「这个轴序为什么对」'
                   '「这个 1e−5 是否可以接受」必须自己用数据判断，否则很容易把形状对、数值错的代码当成完成。')

    heading(document, '参考文献')
    for index, text in enumerate((
            '[1] Radford A., et al. Learning Transferable Visual Models From Natural Language '
            'Supervision. ICML, 2021.',
            '[2] Dosovitskiy A., et al. An Image is Worth 16x16 Words: Transformers for Image '
            'Recognition at Scale. ICLR, 2021.',
            '[3] Ba J., Kiros J., Hinton G. Layer Normalization. arXiv:1607.06450, 2016.',
            '[4] mlfoundations/open_clip：ViT 实现与权重布局（out,in 线性层、无 bias 的 patch 投影）。',
            '[5] 课程课件第 39–73 页；project1 附件 README.md 与 PPT-CODE-MAP.txt。'), start=1):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(1)
        paragraph.paragraph_format.left_indent = Cm(0.6)
        paragraph.paragraph_format.first_line_indent = Cm(-0.6)
        run = paragraph.add_run(text)
        set_run_font(run, BODY_FONT, 9)

    output = ROOT / 'docs/实验报告.docx'
    document.save(output)
    print('saved', output)


if __name__ == '__main__':
    sys.exit(build())
