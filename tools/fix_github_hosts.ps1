# 修复 GitHub 访问：把 github.com 等域名指向实测可用的 IP
#
# 【背景】
# 校园网 DNS 把 github.com 解析到 20.205.243.166，而该 IP 的 443 被阻断，
# 浏览器报 ERR_CONNECTION_TIMED_OUT。实测 github.com 的其他 IP
# （20.27.177.113、140.82.112/113/114.3）可以正常建立 TLS 连接并返回 200，
# 所以改 hosts 即可绕过，不需要任何代理软件。
#
# 【可逆】写入的区块带首尾标记，Remove-GithubHosts.ps1 可整段删除。
#
# 用法（管理员）：
#   powershell -ExecutionPolicy Bypass -File tools\fix_github_hosts.ps1
# 测试（不需要管理员，写到临时文件）：
#   powershell -ExecutionPolicy Bypass -File tools\fix_github_hosts.ps1 -Path .\hosts_test.txt

param(
    [string]$Path = "C:\Windows\System32\drivers\etc\hosts",
    [switch]$Remove,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$Marker = "# --- GitHub access fix (added 2026-10-27) ---"
$EndMarker = "# --- end ---"

# 要写入的映射。IP 均在本机实测通过（TCP 443 可建连 + HTTPS 返回 200/301）。
$Entries = @(
    "20.27.177.113    github.com",
    "140.82.113.6     api.github.com",
    "140.82.113.9     codeload.github.com",
    "185.199.110.133  raw.githubusercontent.com",
    "185.199.110.133  objects.githubusercontent.com",
    "185.199.110.133  gist.githubusercontent.com",
    "185.199.110.133  avatars.githubusercontent.com"
)

$isSystemHosts = ($Path -eq "C:\Windows\System32\drivers\etc\hosts")

if ($isSystemHosts) {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "[错误] 需要管理员权限。请右键 fix_github_hosts.bat -> 以管理员身份运行。" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $Path)) {
    Write-Host "[错误] 找不到 hosts 文件: $Path" -ForegroundColor Red
    exit 1
}

$content = Get-Content $Path -Raw

# -Remove：把带标记的整段删掉，恢复 hosts 原状（只删自己写入的部分）
if ($Remove) {
    if ($content -notmatch [regex]::Escape($Marker)) {
        Write-Host "hosts 中没有本工具写入的记录，无需撤销。"
    } else {
        $inblock = $false
        $out = @()
        foreach ($l in (Get-Content $Path)) {
            if ($l -like "*GitHub access fix*") { $inblock = $true; continue }
            if ($inblock -and $l -like "*# --- end ---*") { $inblock = $false; continue }
            if (-not $inblock) { $out += $l }
        }
        Set-Content -Path $Path -Value $out -Encoding ASCII
        Write-Host "已移除 GitHub 相关记录，hosts 恢复原状。"
    }
    if ($isSystemHosts) { ipconfig /flushdns | Out-Null; Write-Host "DNS 缓存已刷新。" }
    exit 0
}

if ($content -match [regex]::Escape($Marker)) {
    Write-Host "条目已存在，无需重复写入。"
} else {
    $lines = @("", $Marker) + $Entries + @($EndMarker)
    Add-Content -Path $Path -Value $lines -Encoding ASCII
    Write-Host "已写入 $($Entries.Count) 条 GitHub 记录 -> $Path"
}

if ($isSystemHosts) {
    ipconfig /flushdns | Out-Null
    Write-Host "DNS 缓存已刷新。"

    Write-Host ""
    Write-Host "连通性检测："
    foreach ($u in @("https://github.com", "https://api.github.com")) {
        try {
            $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 15 -MaximumRedirection 0 -ErrorAction Stop
            Write-Host ("  OK    {0}  HTTP {1}" -f $u, $r.StatusCode) -ForegroundColor Green
        } catch {
            $resp = $_.Exception.Response
            if ($resp) {
                Write-Host ("  OK    {0}  HTTP {1}" -f $u, [int]$resp.StatusCode) -ForegroundColor Green
            } else {
                Write-Host ("  FAIL  {0}  {1}" -f $u, $_.Exception.Message) -ForegroundColor Yellow
            }
        }
    }
}

Write-Host ""
Write-Host "完成。请关掉浏览器再重开（浏览器有自己的 DNS 缓存），然后访问 https://github.com/signup"
