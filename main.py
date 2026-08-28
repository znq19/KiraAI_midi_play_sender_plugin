"""MIDI演奏语音条插件：搜索MIDI谱 → 合成渲染OGG → 直发QQ语音条。

两入口：
  A. LLM 工具 midi_play_send（自然语言，默认开）
     —— 工具内同步只做"搜谱+下载MIDI"（1~3s），渲染放后台任务；
        完成后直发语音条 + publish_notice 触发 LLM 完成回复（消息合并进会话）。
  B. 命令词钩子（/弹奏 /弹琴，默认关，白名单仅命令式）

对齐 bili_audio_sender：BaseTool 动态注入、白名单仅命令式、工具内直发、
超时容忍、错误可读化、cache_dir + 自动清理。
"""
import asyncio
import os
import re
import shutil
from pathlib import Path

try:
    from . import midi_src, midi_synth
except ImportError:  # 独立运行 / 非包方式加载时回退
    import midi_src  # type: ignore
    import midi_synth  # type: ignore

from core.plugin import BasePlugin, on, Priority
from core.logging_manager import get_logger
from core.utils.tool_utils import BaseTool
from core.provider import LLMRequest
from core.chat import MessageChain
from core.chat.message_utils import KiraMessageEvent
from core.chat.message_elements import Text, Record, File
from core.utils.path_utils import get_data_path

logger = get_logger("midi_play_sender", "green")


class MidiPlayTool(BaseTool):
    name = "midi_play_send"
    description = (
        "搜索MIDI谱并用合成器演奏，发QQ语音条。用户要求弹奏/MIDI演奏/来首钢琴版/八位版等时调用。"
        "song为曲名，建议用英文名或中英混合（如 Canon in D，纯中文可能搜不到谱）；"
        "preset可选乐器(piano/guitar/nylon/organ/8bit/square/strings/bell/flute/default)；"
        "url可选（任意谱子页面/直链/MIDI文件直链/本地路径，或从搜索结果指定，优先于搜索）。"
        "渲染需数秒到数十秒，工具先返回，完成后自动发送语音条。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "song": {"type": "string", "description": "曲名（必填）"},
            "preset": {"type": "string", "description": "乐器preset（可选）"},
            "url": {"type": "string", "description": "已知谱子页面URL（可选，优先于搜索）"},
        },
    }

    def __init__(self, ctx, plugin):
        self.ctx = ctx
        self.plugin = plugin

    async def execute(self, event, song: str = "", preset: str = "", url: str = "",
                      *args, **kwargs) -> str:
        if event.adapter.platform != "QQ":
            return "当前会话不是QQ，无法发送语音条"
        try:
            return await self.plugin._handle_request(event, song or "", preset or "", url or "")
        except midi_src.MidiSrcError as e:
            return str(e)
        except midi_synth.MidiRenderError as e:
            return str(e)
        except Exception as e:
            logger.exception("[midi_play_sender] tool failed")
            # 部分异常 str 为空（如 httpx.ConnectError）：用异常类名兜底，避免"处理失败："空消息
            msg = str(e).strip() or type(e).__name__
            return f"处理失败：{msg}"


class MidiPlaySenderPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        # 默认值与 schema.json 保持一致
        self.enabled = True
        self.enable_tool = True
        self.enable_command = False
        self.command_words = ["/弹奏", "/弹琴"]
        self.default_preset = "piano"
        self.search_count = 5
        self.auto_pick_top = False
        self.max_render_seconds = 600
        self.timeout = 60
        self.score_dir_str = "files/midi_lib"
        self.cache_dir_str = "files/midi_cache"
        self.keep_midi = True
        self.copy_import_scores = True
        self.max_cache_files = 50
        self.cleanup_count = 20
        self.allowed_users: list[str] = []
        self.allowed_groups: list[str] = []
        self.permission_denied_message = "❌ 权限不足：你不在本功能白名单内"
        self.files_dir: Path | None = None   # 渲染缓存（OGG）
        self.scores_dir: Path | None = None  # 谱库（MIDI，默认不清理）
        self.tasks: dict[str, asyncio.Task] = {}

    async def initialize(self):
        sec = self.plugin_cfg.get("section_basic", {})
        self.enabled = sec.get("enabled", True)
        self.enable_tool = sec.get("enable_tool", True)
        self.enable_command = sec.get("enable_command", False)
        self.command_words = [str(w).strip() for w in sec.get("command_words", ["/弹奏", "/弹琴"])
                              if str(w).strip()]
        self.default_preset = sec.get("default_preset", "piano") or "piano"
        if self.default_preset not in midi_synth.PRESETS:
            self.default_preset = "piano"

        perm = self.plugin_cfg.get("section_permission", {})
        self.allowed_users = [str(u).strip() for u in perm.get("allowed_users", []) if str(u).strip()]
        self.allowed_groups = [str(g).strip() for g in perm.get("allowed_groups", []) if str(g).strip()]
        self.permission_denied_message = perm.get(
            "permission_denied_message", "❌ 权限不足：你不在本功能白名单内")

        s = self.plugin_cfg.get("section_search", {})
        self.search_count = max(1, int(s.get("search_count", 5) or 5))
        self.auto_pick_top = s.get("auto_pick_top", False)

        r = self.plugin_cfg.get("section_render", {})
        self.max_render_seconds = max(0, int(r.get("max_render_seconds", 600) or 600))
        self.timeout = max(5, int(r.get("timeout", 60) or 60))

        c = self.plugin_cfg.get("section_cache", {})
        self.score_dir_str = c.get("score_dir", "files/midi_lib") or "files/midi_lib"
        self.cache_dir_str = c.get("cache_dir", "files/midi_cache") or "files/midi_cache"
        self.keep_midi = c.get("keep_midi", True)
        self.copy_import_scores = c.get("copy_import_scores", True)
        self.max_cache_files = max(1, int(c.get("max_cache_files", 50) or 50))
        self.cleanup_count = max(1, int(c.get("cleanup_count", 20) or 20))

        # 谱库与渲染缓存分离：谱库默认永久保留，渲染缓存按上限自动清理
        self.scores_dir = Path(get_data_path()) / self.score_dir_str
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = Path(get_data_path()) / self.cache_dir_str
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._purge_bad_scores()
        self._cleanup_cache()
        logger.info(
            f"[midi_play_sender] ready | tool={'on' if self.enable_tool else 'off'} "
            f"cmd={'on' if self.enable_command else 'off'} preset={self.default_preset} "
            f"max_render={self.max_render_seconds}s keep_midi={'on' if self.keep_midi else 'off'} "
            f"lib={self.score_dir_str} cache={self.cache_dir_str}")

    async def terminate(self):
        for t in list(self.tasks.values()):
            if t and not t.done():
                t.cancel()
        self.tasks.clear()

    # ---------- 权限（仅命令式入口生效；LLM 工具不受白名单限制） ----------
    def _is_allowed_event(self, event) -> bool:
        if not self.allowed_users and not self.allowed_groups:
            return True
        user_id = group_id = ""
        if hasattr(event, "message"):
            msg = event.message
            if msg.sender and getattr(msg.sender, "user_id", None):
                user_id = str(msg.sender.user_id)
            if msg.group:
                group_id = str(getattr(msg.group, "group_id", ""))
        elif getattr(event, "messages", None):
            last = event.messages[-1]
            if last.sender and getattr(last.sender, "user_id", None):
                user_id = str(last.sender.user_id)
            if last.group:
                group_id = str(getattr(last.group, "group_id", ""))
        if self.allowed_users and (not user_id or user_id not in self.allowed_users):
            return False
        if self.allowed_groups and (not group_id or group_id not in self.allowed_groups):
            return False
        return True

    @staticmethod
    def _get_sid(event) -> str:
        sid = getattr(event, "sid", None)
        if sid:
            return sid
        if getattr(event, "session", None) and getattr(event.session, "sid", None):
            return event.session.sid
        last = event.messages[-1]
        adapter = event.adapter.name if event.adapter else "qq"
        if getattr(last, "group", None) and last.group:
            return f"{adapter}:gm:{last.group.group_id}"
        return f"{adapter}:dm:{last.sender.user_id}"

    # ---------- 缓存 ----------
    @staticmethod
    def _slug(s: str) -> str:
        s = re.sub(r'[\\/:*?"<>|\s]+', "_", s).strip("_")
        return s[:40] or "midi"

    def _find_ogg(self, slug: str, preset: str) -> Path | None:
        if self.files_dir is None:
            return None
        p = self.files_dir / f"{slug}_{preset}.ogg"
        return p if p.exists() and p.stat().st_size > 0 else None

    def _find_midi(self, slug: str) -> Path:
        if self.scores_dir is None:
            return Path(f"{slug}.mid")
        return self.scores_dir / f"{slug}.mid"

    def _find_local_score(self, song: str) -> Path | None:
        """本地谱库模糊匹配：文件名（忽略空格/下划线/连字符，转小写）包含歌名即命中。

        支持用户上传的谱子：说"弹奏 Secret Base"能命中 secret_base.mid。
        命中文件做完整解析校验：轻微损坏（data byte 越界）自动修复保留，
        无法修复的坏谱自动清除并跳过（防止坏谱阻塞在线搜索回退）。
        """
        if self.scores_dir is None or not self.scores_dir.exists():
            return None

        def norm(s: str) -> str:
            return re.sub(r"[\s_\-]+", "", s.lower())

        nk = norm(song)
        if not nk:
            return None
        for f in sorted(self.scores_dir.glob("*.mid")):
            if f.stat().st_size <= 0 or nk not in norm(f.stem):
                continue
            try:
                midi_synth.midi_duration(f)
            except midi_synth.MidiRenderError:
                # 先尝试自动修复（个别 data byte 越界的谱可救回），修复失败才清除
                tmp = f.with_suffix(".repair.mid")
                try:
                    if midi_synth.repair_midi_file(f, tmp):
                        tmp.replace(f)
                        logger.warning(f"[midi_play_sender] 谱库坏文件已自动修复: {f.name}")
                    else:
                        logger.warning(f"[midi_play_sender] 谱库坏文件，自动清除: {f}")
                        f.unlink(missing_ok=True)
                        continue
                except Exception:
                    logger.warning(f"[midi_play_sender] 谱库坏文件，自动清除: {f}")
                    f.unlink(missing_ok=True)
                    continue
            return f
        return None

    def _purge_bad_scores(self) -> None:
        """初始化时清理谱库中的非法文件（非 MThd / 空文件）。内容损坏头正确的靠匹配时强校验清除。"""
        if self.scores_dir is None or not self.scores_dir.exists():
            return
        for f in self.scores_dir.glob("*.mid"):
            try:
                if f.stat().st_size > 0 and midi_src.is_valid_midi(f):
                    continue
            except OSError:
                pass
            logger.warning(f"[midi_play_sender] 清理非法谱子: {f}")
            f.unlink(missing_ok=True)

    @staticmethod
    def _resolve_local_path(ref: str) -> Path | None:
        """解析本地 MIDI 路径引用（bot 传的本地路径）。

        支持：绝对路径（C:\\xxx\\a.mid）、data/ 相对路径（自动拼 get_data_path）、
        file:// 协议（file:///C:/xxx/a.mid）。非本地引用（http 开头）返回 None。
        """
        ref = (ref or "").strip().strip('"')
        if not ref or ref.startswith(("http://", "https://")):
            return None
        if ref.startswith("file://"):
            ref = ref[len("file://"):]  # file:///C:/x -> /C:/x
            if re.match(r"^/[A-Za-z]:", ref):
                ref = ref[1:]           # /C:/x -> C:/x
        p = Path(ref)
        if p.is_file() and p.stat().st_size > 0:
            return p
        if ref.startswith("data/"):
            p2 = Path(get_data_path()) / ref[len("data/"):]
            if p2.is_file() and p2.stat().st_size > 0:
                return p2
        return None

    def _import_local_score(self, local: Path, dst: Path) -> Path | None:
        """处理本地/外部获得的谱子：试读 → 解析失败自动修复 → 按 copy_import_scores
        复制入谱库（方便下次点歌直接命中）或直接使用原文件。

        返回实际可用的谱子路径；文件损坏且无法修复时返回 None。
        """
        try:
            midi_synth.midi_duration(local)
            usable = local
        except midi_synth.MidiRenderError:
            # 尝试自动修复（个别 data byte 越界的谱可救回）
            tmp = Path(str(local) + ".repair.mid")
            if not midi_synth.repair_midi_file(local, tmp):
                return None
            usable = tmp
            logger.warning(f"[midi_play_sender] 本地谱已自动修复: {usable.name}")
        if self.copy_import_scores:
            shutil.copy2(usable, dst)
            if usable != local:
                usable.unlink(missing_ok=True)  # 清理修复临时文件，不留脏
            logger.info(f"[midi_play_sender] 本地谱已入谱库: {dst.name}")
            return dst
        # 关闭复制：直接用原文件（不占谱库，但下次可能找不到）
        if usable != local:
            # 修复路径：修复文件是派生产物，仍复制入谱库并清理临时文件，避免泄漏
            shutil.copy2(usable, dst)
            usable.unlink(missing_ok=True)
            logger.info(f"[midi_play_sender] 修复后的谱子已入谱库: {dst.name}")
            return dst
        return usable

    def _cleanup_cache(self) -> None:
        """渲染缓存（OGG）总是按上限清理；谱库（MIDI）默认永久保留（keep_midi=true 不清理）。"""
        self._cleanup_dir(self.files_dir)
        if not self.keep_midi:
            self._cleanup_dir(self.scores_dir)

    def _cleanup_dir(self, d: Path | None) -> None:
        try:
            if d is None or not d.exists():
                return
            files = [f for f in d.iterdir() if f.is_file()]
            if len(files) <= self.max_cache_files:
                return
            files.sort(key=lambda f: f.stat().st_mtime)
            to_delete = files[:self.cleanup_count]
            deleted = 0
            for f in to_delete:
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    logger.warning(f"[midi_play_sender] 清理旧缓存失败 {f}")
            logger.info(f"[midi_play_sender] 缓存清理：删除 {deleted} 个旧文件，剩余 {len(files) - deleted} 个")
        except Exception:
            logger.warning("[midi_play_sender] 缓存清理异常")

    # ---------- 入口 A：LLM 工具注入 ----------
    @on.llm_request(priority=Priority.HIGH)
    async def inject_tool(self, event, req: LLMRequest, *_):
        if not self.enabled or not self.enable_tool:
            return
        try:
            req.tool_set.add(MidiPlayTool(ctx=self.ctx, plugin=self))
        except Exception:
            logger.exception("[midi_play_sender] tool inject failed")

    # ---------- 入口 B：命令词钩子 ----------
    @on.im_message(priority=Priority.HIGH)
    async def handle_command(self, event: KiraMessageEvent, *_):
        if not self.enabled or not self.enable_command or event.adapter.platform != "QQ":
            return
        text = "".join(e.text for e in event.message.chain if isinstance(e, Text)).strip()
        if not text:
            return
        if not any(text == c or text.startswith(c + " ") for c in self.command_words):
            return
        if not self._is_allowed_event(event):
            await self.ctx.message_processor.send_message_chain(
                event.session.sid, MessageChain([Text(self.permission_denied_message)]))
            event.discard(force=True)
            event.stop()
            return
        args = text.split(maxsplit=1)
        target = args[1].strip() if len(args) > 1 else ""
        try:
            reply = await self._handle_request(event, target, "", "")
        except midi_src.MidiSrcError as e:
            reply = f"弹奏失败：{e}"
        except midi_synth.MidiRenderError as e:
            reply = f"弹奏失败：{e}"
        except Exception as e:
            logger.exception("[midi_play_sender] command failed")
            reply = f"弹奏失败：{e}"
        await self.ctx.message_processor.send_message_chain(
            event.session.sid, MessageChain([Text(reply)]))
        event.discard(force=True)
        event.stop()

    # ---------- 入口 C：LLM 唤醒时顺便入库用户发的 MIDI 文件 ----------
    @on.llm_request(priority=Priority.HIGH)
    async def import_midi_on_llm(self, event, req: LLMRequest, *_):
        """LLM 被唤醒（消息进入对话处理）且消息链含 .mid/.midi 文件 → 后台静默入库。

        设计原则：文件入库只在 LLM 确实处理这条消息时发生——消息没唤醒 LLM
        （bot 没被 @ / 闲聊）就不在后台偷跑，避免不可预期的静默行为与存储占用。
        后台任务不阻塞 LLM 请求；入库静默（记日志），LLM 正在回复用户，无需插件确认。
        """
        if not self.enabled:
            return
        try:
            files = []
            for m in getattr(event, "messages", None) or []:
                for e in getattr(m, "chain", None) or []:
                    if (isinstance(e, File)
                            and str(getattr(e, "name", "") or "").lower().endswith((".mid", ".midi"))):
                        files.append(e)
            if files:
                asyncio.create_task(self._import_midi_files(files))
        except Exception:
            logger.exception("[midi_play_sender] import midi files failed")

    async def _import_midi_files(self, files: list) -> None:
        """后台静默入库：有效 MIDI 存入谱库（供'弹奏 xxx'命中），
        轻微损坏（data byte 越界等）先尝试自动修复，救回则入修复版，无法修复才跳过。"""
        saved: list[str] = []
        bad: list[str] = []
        for f in files:
            try:
                src = await f.to_path()
                if not src or not os.path.exists(src):
                    logger.warning(f"[midi_play_sender] 文件不可用: {getattr(f, 'name', '')}")
                    continue
                name = str(f.name or "untitled").rsplit(".", 1)[0]
                tmp = None
                try:
                    midi_synth.midi_duration(src)
                except midi_synth.MidiRenderError:
                    # 与 _find_local_score/_import_local_score 一致：先尝试自动修复再入库
                    tmp = Path(str(src) + ".repair.mid")
                    try:
                        if not midi_synth.repair_midi_file(src, tmp):
                            bad.append(name)
                            continue
                        midi_synth.midi_duration(tmp)  # 修复后必须能完整解析
                        src = tmp
                        logger.warning(f"[midi_play_sender] 收到的坏谱已自动修复: {name}")
                    except Exception:
                        bad.append(name)
                        continue
                target = self.scores_dir / f"{self._slug(name)}.mid"
                shutil.copy2(src, target)
                if tmp is not None:
                    tmp.unlink(missing_ok=True)  # 清理修复临时文件，不留脏
                saved.append(name)
            except Exception:
                logger.exception("[midi_play_sender] 保存MIDI文件失败")
        if saved:
            logger.info(f"[midi_play_sender] LLM 会话中的 MIDI 谱子已静默入库: {saved}")
        if bad:
            logger.warning(f"[midi_play_sender] 非有效 MIDI 未入库: {bad}")

    # ---------- 核心处理 ----------
    async def _handle_request(self, event, song: str, preset: str, url: str) -> str:
        sid = self._get_sid(event)
        song = (song or "").strip()
        url = (url or "").strip()
        if not song and not url:
            return "请提供曲名（如：弹一首 未闻花名）"
        preset = (preset or "").strip() or self.default_preset
        if preset not in midi_synth.PRESETS:
            preset = self.default_preset
        slug = self._slug(song or url)

        # 1) OGG 缓存命中 → 同步直发
        cached_ogg = self._find_ogg(slug, preset)
        if cached_ogg:
            return await self._send_ogg(sid, cached_ogg, song, preset)

        # 本地谱库优先（用户上传/历史下载的谱，文件名模糊匹配）
        mid = self._find_local_score(song) or self._find_midi(slug)

        # 2) 无 MIDI 缓存 → 搜索/下载（同步，快）
        if not (mid.exists() and mid.stat().st_size > 0):
            if url:
                if url.startswith(("http://", "https://")):
                    # 网络 URL：下载（失效则回退按歌名重搜）
                    try:
                        await midi_src.download(url, mid)
                    except midi_src.MidiSrcError as e:
                        if not song:
                            return f"谱子下载失败：{e}"
                        # url 失效（bitmidi 索引可能有过期条目）：回退按歌名重搜，逐个候选尝试
                        logger.warning(f"[midi_play_sender] url 失效({e})，回退搜索「{song}」")
                        ok, err = await midi_src.search_and_download(song, mid, self.search_count)
                        if not ok:
                            return f"谱子下载失败：{err}"
                else:
                    # 本地引用（绝对路径 / data/ 相对 / file://）
                    local = self._resolve_local_path(url)
                    if local is not None:
                        # 试读校验 + 自动修复 + 按配置复制入谱库
                        imported = self._import_local_score(local, mid)
                        if imported is None:
                            return f"本地谱子不可用：{url}（文件损坏且无法自动修复）"
                        mid = imported
                        logger.info(f"[midi_play_sender] 使用本地谱子: {local}")
                    elif song:
                        # 本地文件不存在/已被清理：回退按歌名搜索（绝不把本地路径当网络 URL）
                        logger.warning(f"[midi_play_sender] 本地谱不存在({url})，回退搜索「{song}」")
                        cands = await midi_src.pick_midi(song, self.search_count)
                        if not cands:
                            return f"本地谱不存在，且在线也没找到「{song}」，试试英文名（如 Canon in D）"
                        if self.auto_pick_top:
                            ok, err = await midi_src.search_and_download(song, mid, self.search_count)
                            if not ok:
                                return f"谱子下载失败：{err}"
                        else:
                            return self._format_candidates(song, cands)
                    else:
                        return f"本地谱子不存在或不可用：{url}"
            else:
                cands = await midi_src.pick_midi(song, self.search_count)
                if not cands:
                    return f"没找到「{song}」的MIDI谱，试试英文名（如 Canon in D）或加上歌手名"
                if self.auto_pick_top:
                    ok, err = await midi_src.search_and_download(song, mid, self.search_count)
                    if not ok:
                        return f"谱子下载失败：{err}"
                else:
                    return self._format_candidates(song, cands)
            if not midi_src.is_valid_midi(mid):
                mid.unlink(missing_ok=True)
                return "谱子下载失败或文件无效，换一个候选试试"

        # 3) 时长检查（渲染前）
        try:
            dur = midi_synth.midi_duration(mid)
        except midi_synth.MidiRenderError as e:
            return str(e)
        if self.max_render_seconds and dur > self.max_render_seconds:
            return f"该曲谱时长 {dur:.0f}s 超过上限 {self.max_render_seconds}s，未渲染"

        # 4) 异步渲染：工具快速返回，后台完成渲染+直发+通知
        self._cancel_old_task(sid)
        task = asyncio.create_task(
            self._render_and_notify(sid, song or url, preset, mid, slug, dur))
        self.tasks[sid] = task
        return (f"已找到谱子《{song or url}》，正在合成{preset}演奏版"
                f"（约{self._est_seconds(dur)}秒），完成后自动发送")

    @staticmethod
    def _est_seconds(dur: float) -> int:
        est = int(dur * 0.4 + 2)  # 实测约 0.35~0.4x 实时
        return max(5, min(est, 150))

    def _format_candidates(self, song: str, cands: list[dict]) -> str:
        lines = [f"找到「{song}」的MIDI谱："]
        for i, c in enumerate(cands, 1):
            host = "bitmidi" if "bitmidi.com" in c["url"] else (
                "freemidi" if "freemidi.org" in c["url"] else (
                    "hamienet" if "hamienet.com" in c["url"] else "web"))
            lines.append(f"{i}. {c.get('title', '')}｜{host}｜{c['url']}")
        lines.append("可直接传 url 再次调用直接演奏，或展示给用户选择。")
        return "\n".join(lines)

    def _data_rel(self, p: Path) -> str:
        """返回 data/ 相对路径（pixiv 同款，方便 LLM 复用文件）。不在 data 下则原样返回。"""
        try:
            return f"data/{Path(p).relative_to(Path(get_data_path())).as_posix()}"
        except ValueError:
            return str(p)

    async def _send_ogg(self, sid: str, ogg: Path, song: str, preset: str) -> str:
        chain = MessageChain([Record(record=str(ogg), name=Path(ogg).name)])
        result = await self.ctx.message_processor.send_message_chain(sid, chain)
        err = str(result.err or "") if not result.ok else ""
        if not result.ok and "超时" not in err:
            return f"发送失败：{err}"
        if not result.ok:
            logger.warning(f"[midi_play_sender] send timeout but may have delivered: {err}")
        return (
            f"已直接发送《{song}》{preset}版演奏（缓存），无需再次发送，只需简单回应用户。\n"
            f"发送的文件：{self._data_rel(ogg)}")

    async def _render_and_notify(self, sid: str, song: str, preset: str,
                                 mid: Path, slug: str, duration: float) -> None:
        """后台任务：渲染 → 直发语音条 → publish_notice 触发 LLM 完成回复。"""
        ogg = self.files_dir / f"{slug}_{preset}.ogg"
        try:
            out, _ = await asyncio.to_thread(
                midi_synth.render, str(mid), str(ogg), preset)
        except midi_synth.MidiRenderError as e:
            await self._notify(sid, f"演奏失败：《{song}》{e}")
            return
        except Exception as e:
            logger.exception("[midi_play_sender] render failed")
            await self._notify(sid, f"演奏失败：《{song}》{e}")
            return
        self._cleanup_cache()

        result = await self.ctx.message_processor.send_message_chain(
            sid, MessageChain([Record(record=out, name=Path(out).name)]))
        err = str(result.err or "") if not result.ok else ""
        if not result.ok and "超时" not in err:
            await self._notify(sid, f"发送失败：《{song}》{err}")
            return
        if not result.ok:
            logger.warning(f"[midi_play_sender] send timeout but may have delivered: {err}")

        await self._notify(
            sid,
            f"系统通知：MIDI演奏完成，《{song}》{preset}版语音条已发送到群里，"
            "请用一两句话告知用户演奏完成，不要重复发送语音条。"
            f"音频文件：{self._data_rel(Path(out))}；谱子文件：{self._data_rel(mid)}")

    async def _notify(self, sid: str, text: str) -> None:
        """publish_notice：构造合成事件进主线路，LLM 接话完成回复（消息合并进会话）。"""
        try:
            await self.ctx.publish_notice(sid, MessageChain([Text(text)]), is_mentioned=True)
        except Exception:
            logger.exception("[midi_play_sender] publish_notice failed")

    def _cancel_old_task(self, sid: str) -> None:
        t = self.tasks.pop(sid, None)
        if t and not t.done():
            t.cancel()
