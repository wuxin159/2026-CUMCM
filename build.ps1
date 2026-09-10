# 论文编译脚本
#
# 用法: 在项目任意位置执行 .\build.ps1
#       双击也可运行
#
# 之所以连编两遍: 第一遍生成 .aux 交叉引用表, 第二遍才能把
# 图表编号和页码填对, 只编一遍正文里会留下一堆 "??"。
#
# 路径全部相对本脚本位置推导, 不含任何机器专属路径。

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PaperDir = Join-Path $Root 'paper'
$TexFile = 'main.tex'

if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
    Write-Host '找不到 xelatex。' -ForegroundColor Red
    Write-Host '如果 TeX Live 刚装好, 说明当前终端还持有旧的 PATH, 重开终端或重启电脑即可。'
    exit 1
}

Push-Location $PaperDir
try {
    $args = @('-synctex=1', '-interaction=nonstopmode', '-file-line-error', '-shell-escape', $TexFile)
    foreach ($pass in 1, 2) {
        Write-Host "第 $pass 遍编译..." -ForegroundColor Cyan
        # 每遍都覆盖日志。第一遍必然残留未解析的交叉引用,
        # 若追加写入, 后面统计 undefined 会把第一遍的也算进去。
        & xelatex @args | Out-File -FilePath 'build.log' -Encoding utf8
        if ($LASTEXITCODE -ne 0) {
            Write-Host "第 $pass 遍失败, 错误如下:" -ForegroundColor Red
            Select-String -Path 'build.log' -Pattern '^(.*:\d+:|!)' | Select-Object -First 10 | ForEach-Object { $_.Line }
            Write-Host "完整日志: $(Join-Path $PaperDir 'build.log')"
            exit 1
        }
    }

    $undefined = (Select-String -Path 'build.log' -Pattern 'Reference .* undefined' -AllMatches).Count
    $pdf = Join-Path $PaperDir 'main.pdf'
    $pages = (Select-String -Path 'build.log' -Pattern 'Output written on .*\((\d+) pages?' | Select-Object -Last 1).Matches.Groups[1].Value

    Write-Host ''
    Write-Host '编译成功' -ForegroundColor Green
    Write-Host "  页数     : $pages"
    Write-Host "  未定义引用: $undefined"
    Write-Host "  输出     : $pdf"

    # 清理中间文件, 但保留 synctex.gz, 它是 PDF 与源码互相跳转的索引
    Remove-Item -Path 'main.aux', 'main.log', 'main.out', 'main.toc', 'build.log' -ErrorAction SilentlyContinue
}
finally {
    Pop-Location
}
