"""脚本库客户端：和 manager-server 对接。

脚本的**数据**放在 manager-server（跨任务、跨设备共享、可人工审阅/禁用），
本包只负责**执行**。所以这里是一个很薄的 HTTP 客户端。

manager 侧要实现的三个端点
--------------------------
===================================  ==========================================
``GET  /api/scripts``                查候选。query 参数：``q``（关键词）、
                                     ``app``（包名）、``limit``（默认 5）。
                                     返回 ``[{...}, ...]``
``GET  /api/scripts/{id}``            取完整脚本（含 ``stepsJson``）。
``POST /api/scripts``                存一个新脚本。body 是脚本本体，
                                     返回 ``{"id": 123}``
===================================  ==========================================

**返回要短** —— 查找结果会进模型的 context，20 个候选等于把 prompt 撑爆。
所以默认只给 5 条，每条一行摘要。

环境变量
--------
``MOBILE_USE_MANAGER_URL``（默认 ``http://127.0.0.1:9107``）——
执行面要知道 manager 在哪。这是**唯一**需要外部提供的信息。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_MANAGER_URL = "http://127.0.0.1:9107"
DEFAULT_LIMIT = 5
#: 单条候选摘要的最大长度 —— 进 context 的东西都要短。
MAX_SUMMARY_CHARS = 160


def manager_url() -> str:
    return (os.environ.get("MOBILE_USE_MANAGER_URL") or DEFAULT_MANAGER_URL).rstrip("/")


def _pick(item: dict[str, Any], *keys: str) -> Any:
    """按候选键名依次取值（camelCase 优先，snake_case 兜底）。

    manager 的 JSON 是 camelCase（和它其它接口一致：``assembledPrompt``、
    ``startedAt``…），但脚本库也可能被别的客户端写成 snake_case —— 两种都认，
    省得为一处拼写打架。
    """
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


class MacroStoreError(Exception):
    """脚本库访问失败（manager 不可达、返回体不对等）。"""


@dataclass
class MacroCandidate:
    """一条候选脚本的摘要。"""

    id: int
    name: str
    when_to_use: str = ""
    app_package: str | None = None
    step_count: int = 0
    success_count: int = 0
    last_used_at: str | None = None

    def summary(self) -> str:
        parts = [f"#{self.id} {self.name}"]
        if self.app_package:
            parts.append(self.app_package)
        parts.append(f"{self.step_count} 步")
        if self.success_count:
            parts.append(f"成功 {self.success_count} 次")
        if self.when_to_use:
            parts.append(self.when_to_use)
        return " | ".join(parts)[:MAX_SUMMARY_CHARS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "when_to_use": self.when_to_use,
            "app_package": self.app_package,
            "step_count": self.step_count,
            "success_count": self.success_count,
            "last_used_at": self.last_used_at,
        }


class MacroStore:
    """manager-server 脚本库的客户端。"""

    def __init__(self, base_url: str | None = None, timeout: float = 20.0):
        self.base_url = (base_url or manager_url()).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise MacroStoreError(f"manager 返回 {exc.code}: {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise MacroStoreError(f"连不上 manager（{self.base_url}）：{exc}") from exc
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MacroStoreError(f"manager 返回的不是 JSON：{raw[:200]!r}") from exc

    def find(
        self, query: str = "", app_package: str | None = None, limit: int = DEFAULT_LIMIT
    ) -> list[MacroCandidate]:
        """查候选宏。返回**短摘要**列表。"""
        params = [f"limit={int(limit)}"]
        if query:
            params.append(f"q={urllib.parse.quote(query)}")
        if app_package:
            params.append(f"app={urllib.parse.quote(app_package)}")
        payload = self._request("GET", "/api/scripts?" + "&".join(params))
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        candidates: list[MacroCandidate] = []
        for item in items[: int(limit)]:
            if not isinstance(item, dict) or "id" not in item:
                continue
            candidates.append(
                MacroCandidate(
                    id=int(item["id"]),
                    name=str(item.get("name") or "?"),
                    when_to_use=str(_pick(item, "whenToUse", "when_to_use") or ""),
                    app_package=_pick(item, "appPackage", "app_package"),
                    step_count=int(_pick(item, "stepCount", "step_count") or 0),
                    success_count=int(_pick(item, "successCount", "success_count") or 0),
                    last_used_at=_pick(item, "lastUsedAt", "last_used_at"),
                )
            )
        return candidates

    def get(self, macro_id: int) -> dict[str, Any]:
        """取完整宏（含 steps）。"""
        payload = self._request("GET", f"/api/scripts/{int(macro_id)}")
        if not isinstance(payload, dict):
            raise MacroStoreError("manager 返回的宏不是一个对象")
        # manager 存的是 JSON 文本（``stepsJson``）；这里摊开成步骤数组，
        # 好让 :class:`~minitap.mobile_use.macros.model.Macro` 直接吃。
        steps_json = _pick(payload, "stepsJson", "steps_json")
        if isinstance(steps_json, str) and steps_json.strip():
            try:
                payload["steps"] = json.loads(steps_json)
            except json.JSONDecodeError:
                payload["steps"] = []
        payload.setdefault("device_serial", _pick(payload, "deviceSerial", "device_serial"))
        payload.setdefault("app_package", _pick(payload, "appPackage", "app_package"))
        payload.setdefault("when_to_use", _pick(payload, "whenToUse", "when_to_use"))
        return payload

    def save(self, macro: dict[str, Any]) -> dict[str, Any]:
        """存一个宏。返回 manager 的回执（至少含 ``id``）。

        ⚠️ **发给 manager 的键名必须是 camelCase**（和它其它接口一致：
        ``whenToUse`` / ``appPackage`` / ``deviceSerial``）。而本包内部用
        snake_case（``when_to_use``…），所以**在这里统一转** —— 否则
        Jackson 会安静地把这些字段丢成 null，存出来的宏没有 app、没有描述，
        检索时也匹配不上（这个坑踩过一次）。
        """
        steps = macro.get("steps")
        steps_json = macro.get("stepsJson") or macro.get("steps_json")
        if steps is None and isinstance(steps_json, str):
            try:
                steps = json.loads(steps_json)
            except json.JSONDecodeError:
                steps = []
        if steps is None:
            steps = []
        if not steps_json:
            steps_json = json.dumps(steps, ensure_ascii=False)

        payload = {
            "name": macro.get("name"),
            "whenToUse": _pick(macro, "when_to_use", "whenToUse"),
            "appPackage": _pick(macro, "app_package", "appPackage"),
            "deviceSerial": _pick(macro, "device_serial", "deviceSerial"),
            "stepsJson": steps_json,
            "stepCount": len(steps) or int(macro.get("stepCount") or 0),
            "sourceExecutionId": _pick(macro, "source_execution_id", "sourceExecutionId"),
        }
        response = self._request("POST", "/api/scripts", payload)
        if not isinstance(response, dict):
            raise MacroStoreError("manager 返回的不是对象")
        return response

    def record_usage(self, macro_id: int, ok: bool) -> None:
        """记一次使用结果（成功/失败计数）。失败不阻断宏的执行。"""
        try:
            self._request(
                "POST", f"/api/scripts/{int(macro_id)}/usage?ok={'true' if ok else 'false'}"
            )
        except MacroStoreError:
            pass  # 计数失败不该让宏的执行失败
