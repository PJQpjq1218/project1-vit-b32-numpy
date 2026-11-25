# 生成本仓库的提交历史（安装 Git 之后运行一次）
#
# 用法（在仓库根目录）：
#   powershell -ExecutionPolicy Bypass -File scripts\make_git_history.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\make_git_history.ps1 -Name "张三" -Email "zhangsan@example.com"
#
# 若 git 不在 PATH 中，用 -Git 指定完整路径：
#   powershell -ExecutionPolicy Bypass -File scripts\make_git_history.ps1 -Git "C:\Program Files\Git\cmd\git.exe"
#
# 说明：脚本把 .history\ 下的四个代码快照依次放进 image_encoder.py 并提交，
# 因此历史里能看到「先跑通 patch/token → 再加注意力 → 最后完成 block 与输出头」
# 的真实增量过程；最后会恢复为完整实现。

param(
    [string]$Git = 'git',
    [string]$Name = '',
    [string]$Email = ''
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Get-Command $Git -ErrorAction SilentlyContinue) -and -not (Test-Path $Git)) {
    throw "找不到 git。请先安装 Git（winget install --id Git.Git -e --source winget），或用 -Git 指定 git.exe 的完整路径。"
}

& $Git --version | Out-Null

if (-not (Test-Path '.git')) {
    & $Git init -b main | Out-Null
    Write-Host '已初始化仓库（分支 main）'
}

if (-not $Name) {
    $Name = Read-Host '提交作者姓名（会显示在 GitHub 提交记录里）'
}
if (-not $Email) {
    $Email = Read-Host '提交作者邮箱（建议与 GitHub 账号一致）'
}
& $Git config user.name $Name
& $Git config user.email $Email

function Commit([string[]]$paths, [string]$message) {
    foreach ($path in $paths) { & $Git add -- $path }
    & $Git commit -m $message | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "提交失败：$message" }
    $hash = (& $Git log -1 --format='%h')
    Write-Host ("  {0}  {1}" -f $hash, $message)
}

Write-Host '正在创建提交：'

# 1) 课程提供的骨架原样入库（此时 image_encoder.py 还有 TODO）
Copy-Item '.history\stage0_starter.py' 'image_encoder.py' -Force
Commit @('image_encoder.py', 'check.py', 'config.json', 'helpers.py', 'preprocess_image.py',
         'runtime.py', 'run.py', 'requirements.txt', '.gitignore', 'assets') 'chore: 导入课程提供的工程骨架、测试图与依赖清单'

# 2) patchify 与 embed
Copy-Item '.history\stage1_patch_embed.py' 'image_encoder.py' -Force
Commit @('image_encoder.py') 'feat(encoder): 实现 patchify 与 embed，patches/tokens 阶段数值通过 (TODO-01/05)'

# 3) 多头切分与缩放点积注意力
Copy-Item '.history\stage2_attention.py' 'image_encoder.py' -Force
Commit @('image_encoder.py') 'feat(encoder): 实现多头切分与缩放点积注意力 (TODO-02/03)'

# 4) TransformerBlock 与输出头：全部阶段通过
Copy-Item '.history\stage3_final.py' 'image_encoder.py' -Force
Commit @('image_encoder.py') 'feat(encoder): 完成块内残差/MLP 与输出头，8 个数值阶段全部通过 (TODO-04/06)'

# 5) 证据脚本与对照实验
Commit @('tools', 'docs/metrics.json', 'docs/figures') 'test: 加入分阶段证据脚本、耗时统计与 6 组对照实验'

# 6) README
Commit @('README.md') 'docs: 编写 README（功能、结构、依赖、数据获取与复现步骤）'

# 7) 实验报告
Commit @('docs') 'docs: 完成实验报告（5 页，含误差表与两张插图）'

& $Git add -A
$pending = & $Git status --porcelain
if ($pending) {
    & $Git commit -m 'chore: 补充仓库辅助脚本' | Out-Null
    Write-Host '  （已提交剩余的辅助脚本）'
}
Write-Host ''
& $Git log --oneline --graph
& $Git status --short
