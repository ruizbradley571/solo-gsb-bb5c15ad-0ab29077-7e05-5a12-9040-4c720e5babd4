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
- nonce 保留到期取“配置 TTL 到期”与“信封时间戳最晚验收时刻
  （timestamp+TIMESTAMP_SKEW）”两者中较晚者：NONCE_TTL 小于时间偏差
  时，同一签名信封在整个允许时间窗内（含最晚验收时刻本身）重放仍返回
  409，且保留中的记录照样占容量（503）；信封时间窗结束且保留期届满后
  才允许复用 nonce；
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
        self._orig_skew = main.secure_config.timestamp_skew
        main.secure_config.keys = {KEY_ID: SECRET}
        main.secure_config.nonce_capacity = 4
        main.secure_config.nonce_ttl = 600
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

    # ------------- TTL 小于时间偏差：保留到信封最晚验收时刻之后 -------------

    def test_replay_within_envelope_window_when_ttl_shorter_than_skew(self) -> None:
        # 回归：NONCE_TTL(10s) < 时间偏差(300s)。信封在 t0 签名成功，
        # t0+11 时 nonce 的配置 TTL 已到期，但信封仍处于允许时间窗内
        # （最晚验收 t0+300），重放必须仍返回 409，不得执行第二次。
        main.secure_config.nonce_ttl = 10
        main.secure_config.timestamp_skew = 300
        t0 = 1_000_000.0
        with unittest.mock.patch.object(main.time, "time", return_value=t0):
            envelope = self._envelope(
                dict(PLAN_PAYLOAD), nonce="short-ttl-1", timestamp=int(t0)
            )
            resp1 = self._post(envelope)
            self.assertEqual(resp1.status_code, 200)

        with unittest.mock.patch.object(main.time, "time", return_value=t0 + 11):
            resp2 = self._post(envelope)
            self.assertEqual(resp2.status_code, 409)
            self.assertEqual(resp2.json()["error"], "nonce_replayed")

        # 保留到期时刻被顺延到信封最晚验收时刻（t0+300）：主表与堆索引
        # 使用同一到期时刻，此刻记录仍未被清理。
        with _nonce_lock:
            expiry, _token = _nonce_seen[(KEY_ID, "short-ttl-1")]
            self.assertEqual(expiry, t0 + 300)
            heap_top = _nonce_expiry_index[0]
            self.assertEqual(heap_top[0], t0 + 300)
            self.assertEqual(heap_top[2], (KEY_ID, "short-ttl-1"))

    def test_replay_at_exact_envelope_deadline_still_409(self) -> None:
        # 最晚验收时刻本身（now == timestamp+skew，时间窗判定为仍可验收）
        # 必须拒绝重放：到期时刻等于 now 的记录不得被清理。
        main.secure_config.nonce_ttl = 10
        main.secure_config.timestamp_skew = 300
        t0 = 2_000_000.0
        with unittest.mock.patch.object(main.time, "time", return_value=t0):
            envelope = self._envelope(
                dict(PLAN_PAYLOAD), nonce="short-ttl-deadline", timestamp=int(t0)
            )
            self.assertEqual(self._post(envelope).status_code, 200)

        with unittest.mock.patch.object(
            main.time, "time", return_value=t0 + 300
        ):
            # 时间窗边界：abs(skew) == 300 仍通过时间戳校验，随后命中重放。
            resp = self._post(envelope)
            self.assertEqual(resp.status_code, 409)
            self.assertEqual(resp.json()["error"], "nonce_replayed")
            with _nonce_lock:
                self.assertIn((KEY_ID, "short-ttl-deadline"), _nonce_seen)

    def test_nonce_reusable_after_envelope_window_and_retention_end(self) -> None:
        # 信封时间窗结束且保留期届满（now > timestamp+skew）后，nonce 才
        # 允许复用：旧信封此刻本身已超出时间偏差（401），用同 nonce 构造
        # 的新合法信封（新时间戳、新签名）应能成功。
        main.secure_config.nonce_ttl = 10
        main.secure_config.timestamp_skew = 300
        t0 = 3_000_000.0
        with unittest.mock.patch.object(main.time, "time", return_value=t0):
            old = self._envelope(
                dict(PLAN_PAYLOAD), nonce="reuse-after-window", timestamp=int(t0)
            )
            self.assertEqual(self._post(old).status_code, 200)

        t1 = t0 + 301
        with unittest.mock.patch.object(main.time, "time", return_value=t1):
            # 旧信封：时间窗已关闭，先被时间偏差校验拒绝（401）。
            self.assertEqual(self._post(old).status_code, 401)
            # 同 nonce 的新信封（新时间戳 + 新签名）：旧记录保留期届满后
            # 已在锁内清理，重新占用成功。
            fresh = self._envelope(
                dict(PLAN_PAYLOAD), nonce="reuse-after-window", timestamp=int(t1)
            )
            self.assertEqual(self._post(fresh).status_code, 200)

    def test_envelope_extended_retention_counts_toward_capacity(self) -> None:
        # 顺延到信封最晚验收时刻的记录照样占用容量：表中 4 条记录的配置
        # TTL 均已到期但信封时间窗未结束，新 nonce 不能顶掉它们，应 503；
        # 而其中任一 nonce 的重放仍优先返回 409。
        main.secure_config.nonce_ttl = 10
        main.secure_config.timestamp_skew = 300
        t0 = 4_000_000.0
        with unittest.mock.patch.object(main.time, "time", return_value=t0):
            nonces = [f"win-occupied-{i}" for i in range(
                main.secure_config.nonce_capacity
            )]
            for nonce in nonces:
                resp = self._post(
                    self._envelope(dict(PLAN_PAYLOAD), nonce=nonce, timestamp=int(t0))
                )
                self.assertEqual(resp.status_code, 200)

        with unittest.mock.patch.object(main.time, "time", return_value=t0 + 11):
            # 配置 TTL 已过但信封最晚验收时刻（t0+300）未到：记录全部保留。
            with _nonce_lock:
                self.assertEqual(
                    len(_nonce_seen), main.secure_config.nonce_capacity
                )
            resp_replay = self._post(
                self._envelope(
                    dict(PLAN_PAYLOAD), nonce=nonces[0], timestamp=int(t0)
                )
            )
            self.assertEqual(resp_replay.status_code, 409)
            resp_new = self._post(
                self._envelope(
                    dict(PLAN_PAYLOAD), nonce="win-newcomer", timestamp=int(t0 + 11)
                )
            )
            self.assertEqual(resp_new.status_code, 503)
            self.assertEqual(resp_new.json()["error"], "nonce_capacity_exceeded")

    def test_replay_protection_with_real_clock_small_ttl(self) -> None:
        # 不 mock 时钟的端到端回归：ttl=1s、skew=300s，提交后等待 TTL
        # 过期（1.2s），信封仍新鲜，重放必须 409。
        main.secure_config.nonce_ttl = 1
        main.secure_config.timestamp_skew = 300
        envelope = self._envelope(dict(PLAN_PAYLOAD), nonce="real-clock-short-ttl")
        self.assertEqual(self._post(envelope).status_code, 200)
        time.sleep(1.2)
        resp = self._post(envelope)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"], "nonce_replayed")

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

    def test_sweep_keeps_entry_when_now_equals_expiry(self) -> None:
        # 到期时刻本身仍受保护：expiry == now 时记录与索引条目都保留，
        # now 严格晚于 expiry 后才清理。
        t0 = 50_000.0
        with _nonce_lock:
            status, _ = _reserve_nonce((KEY_ID, "boundary-ttl"), t0)
            self.assertEqual(status, "reserved")
            expiry = _nonce_seen[(KEY_ID, "boundary-ttl")][0]
            self.assertEqual(expiry, t0 + main.secure_config.nonce_ttl)
            _sweep_nonces(expiry)
            self.assertIn((KEY_ID, "boundary-ttl"), _nonce_seen)
            self.assertEqual(len(_nonce_expiry_index), 1)
            _sweep_nonces(expiry + 0.001)
            self.assertNotIn((KEY_ID, "boundary-ttl"), _nonce_seen)
            self.assertEqual(len(_nonce_expiry_index), 0)

    def test_reserve_expiry_extends_to_envelope_deadline(self) -> None:
        # 信封最晚验收时刻晚于 TTL 到期时，保留到期取前者；早于 TTL 到期
        # 时仍取 TTL 到期（两者中较晚者）；同一时刻写入主表与堆索引。
        main.secure_config.nonce_ttl = 10
        now = 60_000.0
        with _nonce_lock:
            status, reservation = _reserve_nonce(
                (KEY_ID, "extended"), now, envelope_deadline=now + 300
            )
            self.assertEqual(status, "reserved")
            self.assertEqual(reservation[0], now + 300)  # type: ignore[index]
            self.assertEqual(_nonce_seen[(KEY_ID, "extended")][0], now + 300)
            self.assertEqual(_nonce_expiry_index[0][0], now + 300)

            status, reservation2 = _reserve_nonce(
                (KEY_ID, "not-extended"), now, envelope_deadline=now + 5
            )
            self.assertEqual(status, "reserved")
            self.assertEqual(reservation2[0], now + 10)  # type: ignore[index]
            self.assertEqual(_nonce_seen[(KEY_ID, "not-extended")][0], now + 10)

    def test_extended_record_holds_capacity_until_envelope_deadline(self) -> None:
        # TTL 到期但信封最晚验收时刻未到时，顺延记录不可被清理也不可顶掉：
        # 写满容量后新 nonce 返回 full；到 envelope_deadline 时刻本身仍占
        # 名额，严格晚于该时刻后才释放名额。
        main.secure_config.nonce_ttl = 10
        main.secure_config.nonce_capacity = 2
        now = 70_000.0
        deadline = now + 300
        try:
            with _nonce_lock:
                for i in range(2):
                    status, _ = _reserve_nonce(
                        (KEY_ID, f"ext-cap-{i}"), now, envelope_deadline=deadline
                    )
                    self.assertEqual(status, "reserved")
                # TTL（now+10）早已过去，但最晚验收时刻未到：不能腾名额。
                status, _ = _reserve_nonce(
                    (KEY_ID, "ext-cap-new"), now + 11, envelope_deadline=deadline
                )
                self.assertEqual(status, "full")
                # 最晚验收时刻本身仍占名额（到期时刻不被清理）。
                status, _ = _reserve_nonce(
                    (KEY_ID, "ext-cap-new"), deadline, envelope_deadline=deadline
                )
                self.assertEqual(status, "full")
                # 严格晚于最晚验收时刻后，旧记录被清理，名额可用。
                status, _ = _reserve_nonce(
                    (KEY_ID, "ext-cap-new"),
                    deadline + 0.001,
                    envelope_deadline=deadline,
                )
                self.assertEqual(status, "reserved")
        finally:
            main.secure_config.nonce_capacity = self._orig_capacity

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
