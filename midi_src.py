"""MIDI 谱搜索与下载（httpx 异步版）。

渠道（2026-08-28 实测）：
  主搜索  : BitMidi 站内搜索（URL 均为真实页面，带数字后缀，实测全 200）
  备选搜索: FreeMidi 搜索页（静态 HTML 可解析，bitmidi 无谱时兜底）
  主下载  : BitMidi 歌曲页 → /uploads/数字.mid
  备选下载: FreeMidi 详情页 → /getter-{id}（需先访问详情页拿 PHPSESSID + Referer）
  已废弃  : DuckDuckGo HTML 端点（202 反爬死透）、bitmidi slug 直拼（不带数字全 404）

CLI 用法（便于独立调试）:
  python midi_src.py search <曲名>
  python midi_src.py download <页面URL> [输出路径]
"""
import asyncio
import contextvars
import functools
import json
import re
import sys
import urllib.parse
from pathlib import Path

import httpx

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BITMIDI_REFERER = "https://bitmidi.com/"
HAMIE_REFERER = "http://www.hamienet.com/"
FREEMIDI_BASE = "https://freemidi.org"
UPLOAD_RE = re.compile(r"/uploads/(\d+\.mid)")
# Cloudflare 间歇性错误码：指数退避重试 2 次（0.8s / 2s）
RETRY_STATUS = {502, 520, 429}
RETRY_DELAYS = (0.8, 2.0)

# 控制 httpx 是否读取环境变量代理（HTTP_PROXY/HTTPS_PROXY）。
# 云端 KiraAI 常配置代理且偶发不稳：网络错误时禁用代理直连重试一次，再切换渠道。
_use_proxy = contextvars.ContextVar("midi_src_use_proxy", default=True)


class MidiSrcError(Exception):
    pass


def _wrap_net_errors(ctx: str):
    """把 httpx 网络/状态异常统一转为可读 MidiSrcError（渠道链可跳过、LLM 可读）。

    策略：
    - 网络层错误（RequestError：Connect/Timeout/Proxy/Read 等）→ **禁用代理直连重试一次**
      （云端代理偶发不稳，直连可能成功），仍失败则转 MidiSrcError；
    - 状态码错误（HTTPStatusError）→ 直接转（重试无意义）；
    - 原始异常消息为空时（如 ConnectError）用异常类名兜底。
    """

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except MidiSrcError:
                raise
            except httpx.RequestError:
                # 网络错误：禁用代理（直连）重试一次
                token = _use_proxy.set(False)
                try:
                    return await fn(*args, **kwargs)
                except MidiSrcError:
                    raise
                except httpx.HTTPError as e2:
                    msg = str(e2).strip() or type(e2).__name__
                    raise MidiSrcError(
                        f"{ctx}网络错误：{msg}（已尝试代理与直连），已切换其他渠道") from e2
                finally:
                    _use_proxy.reset(token)
            except httpx.HTTPStatusError as e:
                msg = str(e).strip() or type(e).__name__
                raise MidiSrcError(f"{ctx}请求失败：{msg}，已尝试其他渠道") from e

        return wrapper

    return deco


async def _get_with_retry(c, url, *, params=None, headers=None):
    """GET 请求 + Cloudflare 间歇错误指数退避重试（502/520/429，0.8s/2s 两次）。

    返回 (resp, text)。最终仍非 200 时抛 MidiSrcError（带可读上下文）。
    重试路径上的响应体会主动 aclose()，避免连接不释放。
    """
    last = None
    for i, delay in enumerate((0, *RETRY_DELAYS)):
        if i:
            await asyncio.sleep(delay)
        resp = await c.get(url, params=params, headers=headers)
        if resp.status_code not in RETRY_STATUS:
            return resp
        last = resp.status_code
        await resp.aclose()  # 释放连接，避免持续报错时连接堆积
    raise MidiSrcError(f"请求失败(HTTP {last}，已重试 {len(RETRY_DELAYS)} 次)：{url}")


def _client(timeout: float = 60.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        follow_redirects=True,
        timeout=timeout,
        trust_env=_use_proxy.get(),
    )


@_wrap_net_errors("BitMidi 下载")
async def bitmidi_download(page_url: str, out_path) -> str:
    """从 BitMidi 歌曲页提取 /uploads/数字.mid 并下载，返回保存路径。

    502/520/429 指数退避重试 2 次；其余非 200 抛 MidiSrcError。
    """
    async with _client() as c:
        c.headers["Referer"] = BITMIDI_REFERER
        page = await _get_with_retry(c, page_url)
        if page.status_code != 200:
            raise MidiSrcError(f"谱页不可用(HTTP {page.status_code})：{page_url}")
        m = UPLOAD_RE.search(page.text)
        if not m:
            raise MidiSrcError(f"页面中未找到 MIDI 下载链接：{page_url}")
        dl_url = f"https://bitmidi.com/uploads/{m.group(1)}"
        async with c.stream("GET", dl_url) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes(8192):
                    f.write(chunk)
    return str(out_path)


@_wrap_net_errors("HamieNET 下载")
async def hamienet_download(page_url: str, out_path) -> str:
    """从 HamieNET 页面 URL 提取 {id}_{name}，拼直链下载（需 UA + Referer）。"""
    m = re.search(r"hamienet\.com/([0-9]+_[A-Za-z0-9_\-]+)\.(?:html?|mid)", page_url)
    if not m:
        raise MidiSrcError(f"无法从 URL 提取 HamieNET id_name：{page_url}")
    dl_url = f"http://www.hamienet.com/{m.group(1)}.mid"
    async with _client() as c:
        c.headers["Referer"] = HAMIE_REFERER
        resp = await c.get(dl_url)
        if resp.status_code != 200:
            raise MidiSrcError(f"HamieNET 下载失败：HTTP {resp.status_code}")
        with open(out_path, "wb") as f:
            f.write(resp.content)
    return str(out_path)


@_wrap_net_errors("BitMidi 搜索")
async def bitmidi_search(query: str, limit: int = 5) -> list[dict]:
    """BitMidi 站内搜索页。502/520/429 指数退避重试 2 次；其余非 200 / 网络错误抛 MidiSrcError（渠道链可跳过）。

    标题优先取 anchor 的 title 属性（真实文件名，如 "Never-Gonna-Give-You-Up-1.mid"），
    无 title 时回退 slug 还原（去掉数字后缀）。
    """
    async with _client() as c:
        c.headers["Referer"] = BITMIDI_REFERER
        resp = await _get_with_retry(c, "https://bitmidi.com/search", params={"q": query})
        if resp.status_code != 200:
            raise MidiSrcError(f"BitMidi 搜索暂时不可用(HTTP {resp.status_code})，已尝试其他渠道")
        text = resp.text
    results: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'<a[^>]*href="/([a-z0-9-]+-mid)"[^>]*title="([^"]*)"', text):
        slug, title = m.group(1), m.group(2)
        if slug in seen:
            continue
        seen.add(slug)
        title = title.strip()
        if title.lower().endswith(".mid"):
            title = title[:-4]
        title = title.replace("-", " ").strip() or slug[:-4].replace("-", " ").title()
        results.append({"title": title, "url": f"https://bitmidi.com/{slug}"})
        if len(results) >= limit:
            break
    return results


@_wrap_net_errors("FreeMidi 搜索")
async def freemidi_search(query: str, limit: int = 5) -> list[dict]:
    """FreeMidi 搜索页（静态 HTML 可解析），返回 [{title, url}]。

    URL 形如 https://freemidi.org/download3-{id}-{slug}，全部真实可下载。
    非 200 / 网络错误抛 MidiSrcError（渠道链可跳过）。
    """
    async with _client() as c:
        resp = await _get_with_retry(c, f"{FREEMIDI_BASE}/search", params={"q": query})
        if resp.status_code != 200:
            raise MidiSrcError(f"FreeMidi 搜索暂时不可用(HTTP {resp.status_code})，已尝试其他渠道")
        text = resp.text
    results: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="(/download3-\d+[^"]*)"', text):
        path = m.group(1)
        if path in seen:
            continue
        seen.add(path)
        # 标题从 URL slug 还原（download3-7673-never-gonna-give-you-up-rick-astley）
        parts = path.split("-", 2)
        title = parts[2].replace("-", " ").title() if len(parts) > 2 else query
        results.append({"title": title, "url": f"{FREEMIDI_BASE}{path}"})
        if len(results) >= limit:
            break
    return results


@_wrap_net_errors("FreeMidi 下载")
async def freemidi_download(page_url: str, out_path) -> str:
    """FreeMidi 详情页 → /getter-{id} 下载 MIDI。

    必须：先访问详情页拿 PHPSESSID cookie，再带 Referer 请求 getter，
    否则返回 500。返回 MThd 头 MIDI 文件路径。
    """
    m = re.search(r"download3-(\d+)", page_url)
    if not m:
        raise MidiSrcError(f"无法从 URL 提取 FreeMidi id：{page_url}")
    mid = m.group(1)
    async with _client() as c:
        # 1) 详情页：拿 session cookie
        page = await _get_with_retry(c, page_url)
        if page.status_code != 200:
            raise MidiSrcError(f"FreeMidi 谱页不可用(HTTP {page.status_code})：{page_url}")
        # 2) getter 下载：必须带 Referer + 同一 session
        resp = await _get_with_retry(
            c, f"{FREEMIDI_BASE}/getter-{mid}",
            headers={"Referer": page_url})
        if resp.status_code != 200:
            raise MidiSrcError(f"FreeMidi 下载失败(HTTP {resp.status_code})：{page_url}")
        if resp.content[:4] != b"MThd":
            raise MidiSrcError(f"FreeMidi 下载内容不是有效 MIDI：{page_url}")
        with open(out_path, "wb") as f:
            f.write(resp.content)
    return str(out_path)


async def pick_midi(song: str, limit: int = 5) -> list[dict]:
    """按优先级收集候选谱（2026-08-28 实测）：

    1. BitMidi 站内搜索（最稳：URL 均为真实页面，带数字后缀）
    2. FreeMidi 搜索（bitmidi 无谱/失败时兜底，如 Megalovania）
    3. HamieNET 直链（保留，URL 需用户/搜索提供）

    已废弃：DDG HTML 端点（202 反爬死透）、bitmidi slug 直拼（不带数字全 404）。

    返回 [{title, url}]。
    """
    cands: list[dict] = []
    try:
        cands += await bitmidi_search(song, limit)
    except MidiSrcError:
        pass
    if not cands:
        # bitmidi 无结果（如 Megalovania）或搜索失败 → FreeMidi 兜底
        try:
            cands += await freemidi_search(song, limit)
        except MidiSrcError:
            pass
    seen: set[str] = set()
    out: list[dict] = []
    for c in cands:
        if c["url"] not in seen:
            seen.add(c["url"])
            out.append(c)
    return out[:limit]


async def try_download_any(cands: list[dict], out_path) -> tuple[str | None, str]:
    """逐个尝试下载候选，返回 (成功保存路径 或 None, 失败原因汇总)。

    下载前先 HEAD 预检状态码：404/410 直接跳过，不逐个撞；
    HEAD 不可用（405/网络异常）时跳过预检直接下载。
    """
    errs: list[str] = []
    for c in cands:
        try:
            # 预检：404/410 直接跳过（省一次无效下载）
            if c["url"].startswith(("http://", "https://")):
                skip = False
                try:
                    async with _client() as pc:
                        head = await pc.head(c["url"])
                    if head.status_code in (404, 410):
                        errs.append(f"{c['url']} 页面不存在(HTTP {head.status_code})")
                        skip = True
                except httpx.HTTPError:
                    pass  # HEAD 不可用，跳过预检直接下载
                if skip:
                    continue
            await download(c["url"], out_path)
            if is_valid_midi(out_path):
                return str(out_path), ""
            Path(out_path).unlink(missing_ok=True)
            errs.append(f"{c['url']} 文件无效")
        except Exception as e:
            errs.append(f"{c['url']} {e}")
    return None, "；".join(errs)


async def search_and_download(song: str, out_path, limit: int = 5) -> tuple[str | None, str]:
    """搜索 + 下载一体：bitmidi 候选全灭（404 预检全跳过/下载全失败）时自动兜底 freemidi。

    返回 (成功保存路径 或 None, 失败原因汇总)。
    """
    cands = await pick_midi(song, limit)
    if not cands:
        return None, f"没找到「{song}」的MIDI谱"
    ok, err = await try_download_any(cands, out_path)
    if ok:
        return ok, ""
    # 候选全灭且全部来自 bitmidi → freemidi 兜底再试一次
    if all("bitmidi.com" in c["url"] for c in cands):
        try:
            fm = await freemidi_search(song, limit)
        except MidiSrcError:
            fm = []
        if fm:
            ok2, err2 = await try_download_any(fm, out_path)
            if ok2:
                return ok2, ""
            err = f"{err}；{err2}"
    return None, err


@_wrap_net_errors("直链下载")
async def direct_download(url: str, out_path) -> str:
    """任意直链/页面兜底下载：

    1) 直接 GET：响应即 MIDI（MThd 文件头）→ 直接保存（Google Drive 直链等适用）；
    2) 响应为 HTML 页面 → 提取页面内 .mid 链接（绝对/相对）再下载。
    502/520/429 指数退避重试 2 次；网络错误抛 MidiSrcError。
    """
    async with _client() as c:
        resp = await _get_with_retry(c, url)
        if resp.status_code != 200:
            raise MidiSrcError(f"下载失败(HTTP {resp.status_code})：{url}")
        data = resp.content
        ctype = resp.headers.get("content-type", "")
        if data[:4] == b"MThd" or "html" not in ctype:
            # 直链命中：内容即 MIDI（或非 HTML 二进制）
            with open(out_path, "wb") as f:
                f.write(data)
            if data[:4] != b"MThd":
                raise MidiSrcError(f"下载内容不是有效 MIDI：{url}")
            return str(out_path)
        # HTML 页面 → 提取 .mid 链接
        text = data.decode("utf-8", errors="ignore")
        mid_url = ""
        m = re.search(r'https?://[^\s"\'<>]+\.mid\b', text)
        if m:
            mid_url = m.group(0)
        else:
            m2 = re.search(r'(?:href|src|data-url|action)="([^"]*\.mid)"', text, re.I)
            if m2:
                mid_url = m2.group(1)
        if not mid_url:
            raise MidiSrcError(f"页面中未找到 MIDI 下载链接：{url}")
        if mid_url.startswith("//"):
            mid_url = "https:" + mid_url
        elif mid_url.startswith("/"):
            base = urllib.parse.urlparse(url)
            mid_url = f"{base.scheme}://{base.netloc}{mid_url}"
        return await direct_download(mid_url, out_path)


async def download(page_url: str, out_path) -> str:
    """按 URL 自动选择源下载：
    bitmidi 歌曲页 → /uploads/ 提取；hamienet 页面 → id_name 直链；
    freemidi download3 页面 → getter 下载；其余一律走 direct_download。
    """
    if "hamienet.com" in page_url:
        return await hamienet_download(page_url, out_path)
    if "bitmidi.com" in page_url:
        return await bitmidi_download(page_url, out_path)
    if "freemidi.org" in page_url:
        return await freemidi_download(page_url, out_path)
    return await direct_download(page_url, out_path)


def is_valid_midi(path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"MThd"
    except OSError:
        return False


async def _main(argv):
    if len(argv) < 2:
        print("用法: midi_src.py <search|download> <参数>")
        return
    cmd, arg = argv[0], argv[1]
    if cmd == "search":
        print(json.dumps(await pick_midi(arg), ensure_ascii=False, indent=1))
    elif cmd == "download":
        out = argv[2] if len(argv) > 2 else "midi_dl.mid"
        path = await download(arg, out)
        print(json.dumps({"path": path, "valid": is_valid_midi(path)}, ensure_ascii=False))
    else:
        print("未知命令: " + cmd)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
