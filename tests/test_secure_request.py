# -*- coding: utf-8 -*-
"""POST /secure-request 的鉴权与 nonce 防重放行为测试。

覆盖：
- 成功路径与原签名/时间窗/业务响应语义不变；
- 重放返回 409 nonce_replayed，且在防重放表已满时 409 仍优先于 503；
- 容量对进行中占位与窗口内已成功 nonce 统一计数；容量已满且无过期项时
  返回 503 nonce_capacity_exceeded（附 capacity），不驱逐未过期 nonce、
  不执行 payload；过期项在同一把锁内清理后可自动腾出名额；
- 过期清理由按到期时间推进的索引驱动：单次预留只从索引头部弹出已到期
  条目，不扫描未过期 nonce；
- payload/业务失败（422）释放请求自身占位，可用同一 nonce 重试；
- 失败释放占位时同步移除到期索引条目：索引规模始终等于主表规模，反复
  提交失败请求不会让索引条目超过 SECURE_REQUEST_NONCE_CAPACITY 无界增长；
- 释放占位时若该记录已过期并被并发请求重新占用，不能误删新记录；
- 索引条目携带占位令牌：同名 nonce 到期后重新占用时，遗留的索引条目
  弹出时不得删除主表中的新占位；
- SECURE_REQUEST_NONCE_TTL 小于时间偏差时，验签通过的 nonce 仍保留到
  该信封时间戳允许的最晚验收时刻之后（最晚验收时刻本身仍拒绝重放），
  信封时间窗结束且保留期届满后同一 nonce 可随新信封复用；
- SECURE_REQUEST_NONCE_CAPACITY 默认 10000 且必须为正整数。

直接运行：python -m unittest discover -s tests
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import os
import threading
import time
import unittest
import unittest.mock
import uuid

from app import main
from app.main import (
    SecureRequestConfig,
    _nonce_expiry_index,
    _nonce_lock,
    _nonce_seen,
    _release_nonce,
    _reserve_nonce,
    _sweep_nonces,
    app,
)

try:
    from starlette.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]

KEY_ID = "test-key"
SECRET = b"0123456789abcdef0123456789abcdef"


def canonical_json(obj: object) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


PLAN_PAYLOAD = {
    "object_length": 256,
    "ranges": [{"type": "interval", "start": 0, "end": 99}],
    "max_gap": 8,
    "fetch_limit": 128,
}


@unittest.skipIf(TestClient is None, "需要安装 starlette（随 fastapi 安装）")
class SecureRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        # 进程级单例：测试中直接注入密钥与极小容量，并清空防重放主表与到期索引。
        self._orig_keys = main.secure_config.keys
        self._orig_capacity = main.secure_config.nonce_capacity
        self._orig_ttl = main.secure_config.nonce_ttl
        main.secure_config.keys = {KEY_ID: SECRET}
        main.secure_config.nonce_capacity = 4
        main.secure_config.nonce_ttl = 600
        with _nonce_lock:
            _nonce_seen.clear()
            _nonce_expiry_index.clear()

    def tearDown(self) -> None:
        main.secure_config.keys = self._orig_keys
        main.secure_config.nonce_capacity = self._orig_capacity
        main.secure_config.nonce_ttl = self._orig_ttl
        with _nonce_lock:
            _nonce_seen.clear()
            _nonce_expiry_index.clear()

    def _envelope(
        self,
        payload: dict,
        *,
        operation: str = "plan",
        nonce: str | None = None,
        timestamp: int | None = None,
        key_id: str = KEY_ID,
        secret: bytes = SECRET,
    ) -> dict:
        envelope = {
            "operation": operation,
            "payload": payload,
            "key_id": key_id,
            "timestamp": int(time.time()) if timestamp is None else timestamp,
            "nonce": uuid.uuid4().hex if nonce is None else nonce,
        }
        envelope["signature"] = hmac.new(
            secret, canonical_json(envelope), hashlib.sha256
        ).hexdigest()
        return envelope

    def _post(self, envelope: dict):
        return self.client.post(
            "/secure-request", json=envelope, headers={}
        )

    def _fill_table(self, count: int, *, ttl: float = 300.0) -> list[str]:
        """写入 count 条到期时间约为 now+ttl 的记录，返回占用的 nonce。

        通过正式预留接口（按过去时刻预留）写入，保证主表与到期索引
        同步维护，与真实请求路径一致。
        """
        nonces = [f"occupied-{i}-{uuid.uuid4().hex}" for i in range(count)]
        now = time.time()
        with _nonce_lock:
            for nonce in nonces:
                status, _ = _reserve_nonce(
                    (KEY_ID, nonce), now + ttl - main.secure_config.nonce_ttl
                )
                self.assertEqual(status, "reserved")
        return nonces

    # ------------------------------- 基本语义 -------------------------------

    def test_success_then_replay_409(self) -> None:
        envelope = self._envelope(dict(PLAN_PAYLOAD), nonce="once-1")
        resp1 = self._post(envelope)
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp1.json()["object_length"], 256)

        resp2 = self._post(envelope)
        self.assertEqual(resp2.status_code, 409)
        self.assertEqual(resp2.json()["error"], "nonce_replayed")
        self.assertEqual(resp2.json()["key_id"], KEY_ID)

    def test_bad_signature_rejected_without_occupying_nonce(self) -> None:
        envelope = self._envelope(dict(PLAN_PAYLOAD), nonce="bad-sig-1")
        envelope["signature"] = "0" * 64
        resp = self._post(envelope)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "invalid_signature")
        with _nonce_lock:
            self.assertNotIn((KEY_ID, "bad-sig-1"), _nonce_seen)

    # ----------------------------- 容量与 503 ------------------------------

    def test_capacity_full_returns_503_and_does_not_execute(self) -> None:
        self._fill_table(main.secure_config.nonce_capacity)
        self.assertEqual(len(_nonce_seen), 4)

        executed = threading.Event()

        def _boom(_req):  # pragma: no cover - 容量满时绝不能被调用
            executed.set()
            raise AssertionError("容量已满时不得执行 payload")

        orig_plan = main.plan
        main.plan = _boom
        try:
            resp = self._post(self._envelope(dict(PLAN_PAYLOAD)))
        finally:
            main.plan = orig_plan
        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["error"], "nonce_capacity_exceeded")
        self.assertEqual(body["capacity"], 4)
        self.assertFalse(executed.is_set())
        # 不得驱逐任何未过期 nonce
        self.assertEqual(len(_nonce_seen), 4)

    def test_replay_409_takes_precedence_over_capacity_503(self) -> None:
        nonces = self._fill_table(main.secure_config.nonce_capacity)
        # 表已满，但其中一条 nonce 被重放：必须返回 409 而不是 503。
        resp = self._post(self._envelope(dict(PLAN_PAYLOAD), nonce=nonces[0]))
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "nonce_replayed")

    def test_expired_entries_swept_free_capacity(self) -> None:
        # 写满容量，但全部已过期；下一条合法请求应在同锁内清理后成功。
        self._fill_table(main.secure_config.nonce_capacity, ttl=-1.0)
        self.assertEqual(len(_nonce_seen), 4)

        resp = self._post(self._envelope(dict(PLAN_PAYLOAD), nonce="fresh-1"))
        self.assertEqual(resp.status_code, 200)
        with _nonce_lock:
            self.assertIn((KEY_ID, "fresh-1"), _nonce_seen)
            # 旧记录全部被清理，只剩本次成功的一条
            self.assertEqual(len(_nonce_seen), 1)

    def test_in_flight_placeholder_counts_toward_capacity(self) -> None:
        # 预置 1 条已成功记录，容量为 4；请求 A 在执行 plan 期间持有 1 个
        # 占位，因此其余 2 个名额被直接写入占满后，第 4 个新 nonce 应 503。
        self._fill_table(2)
        entered = threading.Event()
        gate = threading.Event()
        orig_plan = main.plan

        def _blocking_plan(req):
            entered.set()
            self.assertTrue(gate.wait(3.0), "测试同步事件超时")
            return orig_plan(req)

        main.plan = _blocking_plan
        result: dict[str, object] = {}

        def _call_a() -> None:
            result["resp"] = self._post(
                self._envelope(dict(PLAN_PAYLOAD), nonce="in-flight-A")
            )

        try:
            thread = threading.Thread(target=_call_a)
            thread.start()
            self.assertTrue(entered.wait(3.0), "请求 A 未进入执行阶段")
            # A 此时持有占位：连同预置 2 条共 3 条，再加 1 条即满。
            with _nonce_lock:
                self.assertEqual(len(_nonce_seen), 3)
            self._fill_table(1)

            blocked = self._post(self._envelope(dict(PLAN_PAYLOAD), nonce="late-X"))
            self.assertEqual(blocked.status_code, 503)
            self.assertEqual(blocked.json()["error"], "nonce_capacity_exceeded")

            gate.set()
            thread.join(3.0)
        finally:
            main.plan = orig_plan
            gate.set()

        self.assertEqual(result["resp"].status_code, 200)  # type: ignore[attr-defined]
        # A 成功后占位转为正式占用，表仍满（4 条均未过期），late-X 仍 503。
        self.assertEqual(
            self._post(self._envelope(dict(PLAN_PAYLOAD), nonce="late-X")).status_code,
            503,
        )
        # 一条旧记录到期后被下一次请求在同一把锁内清理，腾出名额；
        # 此前被 503 拒绝的 nonce 可重试成功。
        with _nonce_lock:
            for key in list(_nonce_seen):
                if key != (KEY_ID, "in-flight-A"):
                    _expiry, token = _nonce_seen[key]
                    past = time.time() - 1
                    # 主表与到期索引同步修改（旧索引条目成为遗留项，
                    # 弹出时会因记录不匹配被丢弃）
                    _nonce_seen[key] = (past, token)
                    heapq.heappush(_nonce_expiry_index, (past, token, key))
                    break
        retry = self._post(self._envelope(dict(PLAN_PAYLOAD), nonce="late-X"))
        self.assertEqual(retry.status_code, 200)

    # --------------------------- 失败回滚占位 -----------------------------

    def test_invalid_payload_releases_placeholder(self) -> None:
        bad_payload = dict(PLAN_PAYLOAD, object_length=-1)
        envelope = self._envelope(bad_payload, nonce="retry-bad-payload")
        resp1 = self._post(envelope)
        self.assertEqual(resp1.status_code, 422)
        # 占位已释放：同一信封再来一次仍是 422（payload 错误），而非 409。
        resp2 = self._post(envelope)
        self.assertEqual(resp2.status_code, 422)

        # 客户端修正 payload 后用同一 nonce + 新签名重试，应当成功。
        fixed = self._envelope(dict(PLAN_PAYLOAD), nonce="retry-bad-payload")
        resp3 = self._post(fixed)
        self.assertEqual(resp3.status_code, 200)
        # 成功后再重放则 409。
        self.assertEqual(self._post(fixed).status_code, 409)

    def test_business_failure_releases_placeholder(self) -> None:
        bad_business = dict(PLAN_PAYLOAD, object_length=10)  # end 越界
        envelope = self._envelope(bad_business, nonce="retry-business")
        resp1 = self._post(envelope)
        self.assertEqual(resp1.status_code, 422)
        self.assertEqual(resp1.json()["error"], "unsatisfiable")
        resp2 = self._post(envelope)
        self.assertEqual(resp2.status_code, 422)
        self.assertEqual(resp2.json()["error"], "unsatisfiable")

    # ------------------- 释放占位不得误删并发重占记录 -----------------------

    def test_release_keeps_record_reoccupied_after_expiry(self) -> None:
        now = 10_000.0
        main.secure_config.nonce_ttl = 600
        status, old = _reserve_nonce((KEY_ID, "raced"), now)
        self.assertEqual(status, "reserved")
        self.assertIsNotNone(old)

        # 旧请求尚未结束，时间推进到其占位过期；并发请求清理后重新占用。
        later = now + 601
        status, new = _reserve_nonce((KEY_ID, "raced"), later)
        self.assertEqual(status, "reserved")
        self.assertNotEqual(old, new)

        # 旧请求此刻才失败并回滚自己的占位：新记录必须保留。
        with _nonce_lock:
            _release_nonce((KEY_ID, "raced"), old)  # type: ignore[arg-type]
            self.assertEqual(_nonce_seen.get((KEY_ID, "raced")), new)

    def test_release_keeps_record_with_same_expiry_but_new_token(self) -> None:
        now = 20_000.0
        status, old = _reserve_nonce((KEY_ID, "raced2"), now)
        self.assertEqual(status, "reserved")
        # 极端情形：到期时间相同，仅占位令牌不同（记录被重新占用）。
        with _nonce_lock:
            _nonce_seen[(KEY_ID, "raced2")] = (old[0], old[1] ^ 0xDEADBEEF)  # type: ignore[index]
            _release_nonce((KEY_ID, "raced2"), old)  # type: ignore[arg-type]
            self.assertIn((KEY_ID, "raced2"), _nonce_seen)

    # ------------------- 到期索引：只处理到期项、遗留项不误删 -------------------

    def test_sweep_processes_only_expired_entries(self) -> None:
        # 2 条已过期 + 2 条未过期：清理只弹出到期项，未过期记录与其索引
        # 条目都保持不动（无需扫描全表）。
        expired = self._fill_table(2, ttl=-1.0)
        fresh = self._fill_table(2, ttl=300.0)
        with _nonce_lock:
            self.assertEqual(len(_nonce_expiry_index), 4)
            _sweep_nonces(time.time())
            self.assertEqual(len(_nonce_seen), 2)
            for nonce in fresh:
                self.assertIn((KEY_ID, nonce), _nonce_seen)
            for nonce in expired:
                self.assertNotIn((KEY_ID, nonce), _nonce_seen)
            # 索引中只剩下两条未过期记录的条目
            self.assertEqual(len(_nonce_expiry_index), 2)
            self.assertTrue(
                all(expiry > time.time() for expiry, _t, _k in _nonce_expiry_index)
            )

    def test_sweep_stale_index_entry_keeps_reoccupied_record(self) -> None:
        ttl = main.secure_config.nonce_ttl
        t0 = 30_000.0
        status, old = _reserve_nonce((KEY_ID, "idx-raced"), t0)
        self.assertEqual(status, "reserved")
        # 旧占位到期并被并发请求在同一把锁内清理后，同名 nonce 被重新占用。
        t1 = t0 + ttl + 1
        status, new = _reserve_nonce((KEY_ID, "idx-raced"), t1)
        self.assertEqual(status, "reserved")
        self.assertNotEqual(old, new)
        # 模拟旧一代遗留的索引条目（到期时间早于新记录）：弹出时令牌与主表
        # 记录不符，必须直接丢弃，不得删除新占位。
        with _nonce_lock:
            heapq.heappush(_nonce_expiry_index, (old[0], old[1], (KEY_ID, "idx-raced")))  # type: ignore[index]
            _sweep_nonces(t0 + ttl + 0.5)  # 早于新记录的到期时间 t1+ttl
            self.assertEqual(_nonce_seen.get((KEY_ID, "idx-raced")), new)
            # 新占位自己的索引条目仍在堆中，未被误删
            self.assertEqual(len(_nonce_expiry_index), 1)

    def test_release_reclaims_index_entry_immediately(self) -> None:
        # 失败释放占位时，主表记录与到期索引条目一并移除：索引不会滞留到
        # TTL 结束，规模始终与主表一致。
        ttl = main.secure_config.nonce_ttl
        t0 = 40_000.0
        status, reservation = _reserve_nonce((KEY_ID, "idx-released"), t0)
        self.assertEqual(status, "reserved")
        with _nonce_lock:
            self.assertEqual(len(_nonce_expiry_index), 1)
            _release_nonce((KEY_ID, "idx-released"), reservation)  # type: ignore[arg-type]
            self.assertNotIn((KEY_ID, "idx-released"), _nonce_seen)
            # 索引条目随释放立即回收，无需等到期清理
            self.assertEqual(len(_nonce_expiry_index), 0)
            _sweep_nonces(t0 + ttl + 1)
            self.assertEqual(len(_nonce_expiry_index), 0)
            self.assertNotIn((KEY_ID, "idx-released"), _nonce_seen)

    def test_repeated_failures_do_not_bloat_expiry_index(self) -> None:
        # 回归：窗口内反复提交有效签名但 payload 非法的请求，每次都会预留
        # 再回滚；失效索引必须随回滚及时回收，索引规模不随失败次数增长。
        bad_payload = dict(PLAN_PAYLOAD, object_length=-1)
        for _ in range(3 * main.secure_config.nonce_capacity):
            resp = self._post(self._envelope(bad_payload))
            self.assertEqual(resp.status_code, 422)
        with _nonce_lock:
            self.assertEqual(len(_nonce_seen), 0)
            self.assertEqual(len(_nonce_expiry_index), 0)

        # 夹杂成功请求时同样成立：索引规模始终等于主表规模（不超过容量）。
        for i in range(main.secure_config.nonce_capacity - 1):
            resp = self._post(self._envelope(dict(PLAN_PAYLOAD), nonce=f"ok-{i}"))
            self.assertEqual(resp.status_code, 200)
            resp = self._post(self._envelope(bad_payload))
            self.assertEqual(resp.status_code, 422)
        with _nonce_lock:
            self.assertEqual(len(_nonce_seen), main.secure_config.nonce_capacity - 1)
            self.assertEqual(
                len(_nonce_expiry_index), main.secure_config.nonce_capacity - 1
            )
        # 主表未满（失败未占用名额），新 nonce 仍可成功；成功后才计满容量。
        resp = self._post(self._envelope(dict(PLAN_PAYLOAD), nonce="ok-final"))
        self.assertEqual(resp.status_code, 200)
        with _nonce_lock:
            self.assertEqual(len(_nonce_seen), main.secure_config.nonce_capacity)
            self.assertEqual(len(_nonce_expiry_index), main.secure_config.nonce_capacity)
        # 容量满后失败请求仍按 503 拒绝、不驱逐未过期记录，索引也不增长。
        resp = self._post(self._envelope(bad_payload))
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error"], "nonce_capacity_exceeded")
        with _nonce_lock:
            self.assertEqual(len(_nonce_seen), main.secure_config.nonce_capacity)
            self.assertEqual(len(_nonce_expiry_index), main.secure_config.nonce_capacity)


@unittest.skipIf(TestClient is None, "需要安装 starlette（随 fastapi 安装）")
class SecureRequestNonceRetentionTests(unittest.TestCase):
    """SECURE_REQUEST_NONCE_TTL 小于时间偏差时的重放修复回归测试。

    配置 nonce_ttl=60 远小于 timestamp_skew=300：只按 TTL 保留 nonce
    会在信封仍处于允许时间窗时放过重放；修复后保留期延长到该信封的
    最晚验收时刻（timestamp + skew），窗口内（含最晚验收时刻本身）
    一律 409，时间窗结束且保留期届满后 nonce 才可复用。
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        # 进程级单例：注入密钥、时间偏差与（故意偏小的）nonce TTL，
        # 并清空防重放主表与到期索引。
        self._orig_keys = main.secure_config.keys
        self._orig_capacity = main.secure_config.nonce_capacity
        self._orig_ttl = main.secure_config.nonce_ttl
        self._orig_skew = main.secure_config.timestamp_skew
        main.secure_config.keys = {KEY_ID: SECRET}
        main.secure_config.nonce_capacity = 4
        main.secure_config.nonce_ttl = 60
        main.secure_config.timestamp_skew = 300
        with _nonce_lock:
            _nonce_seen.clear()
            _nonce_expiry_index.clear()

    def tearDown(self) -> None:
        main.secure_config.keys = self._orig_keys
        main.secure_config.nonce_capacity = self._orig_capacity
        main.secure_config.nonce_ttl = self._orig_ttl
        main.secure_config.timestamp_skew = self._orig_skew
        with _nonce_lock:
            _nonce_seen.clear()
            _nonce_expiry_index.clear()

    def _envelope(
        self,
        payload: dict,
        *,
        nonce: str | None = None,
        timestamp: int | None = None,
    ) -> dict:
        envelope = {
            "operation": "plan",
            "payload": payload,
            "key_id": KEY_ID,
            "timestamp": int(time.time()) if timestamp is None else timestamp,
            "nonce": uuid.uuid4().hex if nonce is None else nonce,
        }
        envelope["signature"] = hmac.new(
            SECRET, canonical_json(envelope), hashlib.sha256
        ).hexdigest()
        return envelope

    def _post(self, envelope: dict, *, at: float | None = None):
        """提交信封；at 给出时把服务端时钟固定到该时刻。"""
        if at is None:
            return self.client.post("/secure-request", json=envelope, headers={})
        with unittest.mock.patch("app.main.time.time", return_value=at):
            return self.client.post("/secure-request", json=envelope, headers={})

    def test_replay_blocked_until_envelope_deadline_when_ttl_below_skew(self) -> None:
        t0 = 1_000_000.0
        envelope = self._envelope(
            dict(PLAN_PAYLOAD), nonce="ttl-fix", timestamp=int(t0)
        )
        resp = self._post(envelope, at=t0)
        self.assertEqual(resp.status_code, 200)

        # 到期取 max(t0+TTL= t0+60, t0+skew= t0+300)，主表与堆索引一致。
        with _nonce_lock:
            expiry, token = _nonce_seen[(KEY_ID, "ttl-fix")]
            self.assertIn(
                (expiry, token, (KEY_ID, "ttl-fix")), _nonce_expiry_index
            )
        self.assertEqual(expiry, t0 + 300)

        # 配置 TTL 已过但信封仍在时间窗内：同一信封重放必须 409，
        # 而不是第二次执行（修复前此处 nonce 已被清理，会再次 200）。
        resp = self._post(envelope, at=t0 + 120)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "nonce_replayed")

        # 最晚验收时刻本身仍拒绝重放（到期判定为严格小于）。
        resp = self._post(envelope, at=t0 + 300)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "nonce_replayed")
        with _nonce_lock:
            self.assertIn((KEY_ID, "ttl-fix"), _nonce_seen)

        # 时间窗结束：旧信封先因时间偏差被拒（401），与 nonce 无关。
        resp = self._post(envelope, at=t0 + 301)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"], "timestamp_skew")

        # 保留期届满后 nonce 允许复用：新时间戳 + 新签名的信封执行成功。
        reused = self._envelope(
            dict(PLAN_PAYLOAD), nonce="ttl-fix", timestamp=int(t0 + 301)
        )
        resp = self._post(reused, at=t0 + 301)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["object_length"], 256)
        # 复用后该 nonce 重新进入防重放保护。
        self.assertEqual(self._post(reused, at=t0 + 301).status_code, 409)

    def test_ttl_larger_than_skew_keeps_configured_ttl(self) -> None:
        main.secure_config.nonce_ttl = 600
        t0 = 2_000_000.0
        envelope = self._envelope(
            dict(PLAN_PAYLOAD), nonce="ttl-long", timestamp=int(t0)
        )
        resp = self._post(envelope, at=t0)
        self.assertEqual(resp.status_code, 200)
        with _nonce_lock:
            expiry, _token = _nonce_seen[(KEY_ID, "ttl-long")]
        self.assertEqual(expiry, t0 + 600)

    def test_reserve_retention_floor_drives_table_and_index(self) -> None:
        with _nonce_lock:
            status, reservation = _reserve_nonce(
                (KEY_ID, "direct"), 100.0, retain_until=500.0
            )
            self.assertEqual(status, "reserved")
            # 到期取 TTL（100+60=160）与 retain_until（500）中较晚者。
            self.assertEqual(reservation[0], 500.0)
            self.assertEqual(_nonce_seen[(KEY_ID, "direct")], reservation)
            self.assertIn(
                (500.0, reservation[1], (KEY_ID, "direct")), _nonce_expiry_index
            )

            # retain_until 早于 TTL 到期时仍按配置 TTL。
            status2, reservation2 = _reserve_nonce(
                (KEY_ID, "direct2"), 100.0, retain_until=50.0
            )
            self.assertEqual(status2, "reserved")
            self.assertEqual(reservation2[0], 160.0)

            # 到期时刻（500）本身保留，之后才清理（到期 160 的一项一并清理）。
            _sweep_nonces(500.0)
            self.assertIn((KEY_ID, "direct"), _nonce_seen)
            self.assertNotIn((KEY_ID, "direct2"), _nonce_seen)
            self.assertEqual(len(_nonce_expiry_index), 1)
            _sweep_nonces(500.0 + 1e-9)
            self.assertNotIn((KEY_ID, "direct"), _nonce_seen)
            self.assertEqual(len(_nonce_expiry_index), 0)

    def test_window_entries_held_for_capacity_and_409_takes_precedence(self) -> None:
        # 4 条记录的到期时刻为各自信封的最晚验收时刻 400..403
        # （TTL=60 远早于此），表写满容量 4。
        with _nonce_lock:
            for i in range(4):
                status, _ = _reserve_nonce(
                    (KEY_ID, f"keep-{i}"), now=100.0 + i, retain_until=400.0 + i
                )
                self.assertEqual(status, "reserved")
            self.assertEqual(len(_nonce_seen), 4)

        # 最晚验收时刻 400 本身：无任何可清理项，容量仍满 → 新 nonce 503。
        fresh = self._envelope(
            dict(PLAN_PAYLOAD), nonce="newcomer", timestamp=100
        )
        self.assertEqual(self._post(fresh, at=400.0).status_code, 503)

        # 同一时刻重放 keep-0（信封 timestamp=100，最晚验收时刻 400）：
        # 409 优先于 503。
        replay = self._envelope(
            dict(PLAN_PAYLOAD), nonce="keep-0", timestamp=100
        )
        resp = self._post(replay, at=400.0)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "nonce_replayed")

        # 时刻 402：keep-0/keep-1 已过最晚验收时刻，被同锁清理腾名额；
        # keep-2 的最晚验收时刻（402）本身仍保留，重放返回 409。
        fresh2 = self._envelope(
            dict(PLAN_PAYLOAD), nonce="newcomer", timestamp=102
        )
        self.assertEqual(self._post(fresh2, at=402.0).status_code, 200)
        resp = self._post(
            self._envelope(dict(PLAN_PAYLOAD), nonce="keep-2", timestamp=102),
            at=402.0,
        )
        self.assertEqual(resp.status_code, 409)
        with _nonce_lock:
            self.assertIn((KEY_ID, "keep-2"), _nonce_seen)
            self.assertNotIn((KEY_ID, "keep-0"), _nonce_seen)
            self.assertNotIn((KEY_ID, "keep-1"), _nonce_seen)


class NonceCapacityConfigTests(unittest.TestCase):
    """SECURE_REQUEST_NONCE_CAPACITY 的默认值与启动期校验。"""

    ENV_VARS = (
        "SECURE_REQUEST_KEYS",
        "SECURE_REQUEST_TIMESTAMP_SKEW",
        "SECURE_REQUEST_NONCE_TTL",
        "SECURE_REQUEST_NONCE_CAPACITY",
    )

    def setUp(self) -> None:
        self._saved = {k: os.environ.pop(k, None) for k in self.ENV_VARS}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_default_capacity_is_10000(self) -> None:
        cfg = SecureRequestConfig()
        self.assertEqual(cfg.nonce_capacity, 10000)

    def test_custom_capacity_is_parsed(self) -> None:
        os.environ["SECURE_REQUEST_NONCE_CAPACITY"] = "123"
        self.assertEqual(SecureRequestConfig().nonce_capacity, 123)

    def test_capacity_must_be_positive_integer(self) -> None:
        for bad in ("0", "-1"):
            os.environ["SECURE_REQUEST_NONCE_CAPACITY"] = bad
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    SecureRequestConfig()
        os.environ["SECURE_REQUEST_NONCE_CAPACITY"] = "not-an-int"
        with self.assertRaises(ValueError):
            SecureRequestConfig()


if __name__ == "__main__":
    unittest.main()
