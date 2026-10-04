"""mac 节日日历源（v0.19.4 F16）——EventKit 读系统日历的节假日。

用户在日历 app 开启「中国节假日」订阅日历后，当天节日名可直查（春节/
中秋等农历节日拿到的是正确公历日期——内置固定日期表做不到）。TCC 授权
弹一次；拒绝/未开订阅日历/缺 EventKit 绑定全程静默，proactive 回落内置
表。win 不适用（adapter 不注入，走基类 None）。

平台库-free 原则：本模块只被 mac adapter 引用（同 sensor_mac/mouse_lock_mac）。
"""

from __future__ import annotations

import datetime
import logging
import re

_log = logging.getLogger("pet")

# 节假日订阅日历标题特征（中/英文环境各覆盖）
_CAL_KEYWORDS = ("节假日", "节日", "假日", "holidays", "holiday",
                 "festival", "feiertage")
# 事件标题带这些标记的是调休工作日，不是假日
_WORK_MARKS = ("班",)
# 标题尾部括注（如「（休）」）剥掉再当节日名
_STRIP_PAREN = re.compile(r"[（(][^（）()]*[)）]\s*$")


class MacFestivalSource:
    """today_name() → 今天节日名 | None。按天缓存；未授权/无命中返 None。"""

    def __init__(self) -> None:
        self._store = None
        self._granted = False
        self._cache_day: datetime.date | None = None
        self._cache_name: str | None = None
        try:
            from EventKit import (
                EKAuthorizationStatusAuthorized,
                EKEntityTypeEvent,
                EKEventStore,
            )

            status = EKEventStore.authorizationStatusForEntityType_(
                EKEntityTypeEvent)
            self._store = EKEventStore.alloc().init()
            if int(status) == int(EKAuthorizationStatusAuthorized):
                self._granted = True
                return
            # NotDetermined → 请求授权（completion 在 GCD 队列，只置标志）
            def _cb(granted, _error) -> None:
                self._granted = bool(granted)
                _log.info("日历授权%s，节日源%s", "通过" if granted else "被拒",
                          "启用" if granted else "回落内置表")

            self._store.requestAccessToEntityType_completion_(
                EKEntityTypeEvent, _cb)
        except Exception:
            _log.info("EventKit 初始化失败，节日源回落内置表", exc_info=True)
            self._store = None

    def today_name(self) -> str | None:
        today = datetime.date.today()
        if self._cache_day == today:
            return self._cache_name
        self._cache_day = today
        self._cache_name = self._query(today)
        return self._cache_name

    # ---- 内部 ----

    def _query(self, today: datetime.date) -> str | None:
        if not self._granted or self._store is None:
            return None
        try:
            from Foundation import NSDate

            start_dt = datetime.datetime.combine(today, datetime.time.min)
            end_dt = start_dt + datetime.timedelta(days=1)
            ns_start = NSDate.dateWithTimeIntervalSince1970_(
                start_dt.timestamp())
            ns_end = NSDate.dateWithTimeIntervalSince1970_(end_dt.timestamp())
            pred = self._store.predicateForEventsWithStartDate_endDate_calendars_(
                ns_start, ns_end, None)
            for ev in self._store.eventsMatchingPredicate_(pred) or ():
                cal = ev.calendar()
                cal_title = ((cal.title() if cal is not None else "") or "")
                if not any(k in cal_title.lower() for k in _CAL_KEYWORDS):
                    continue
                title = _STRIP_PAREN.sub("", (ev.title() or "").strip()).strip()
                if not title or any(m in title for m in _WORK_MARKS):
                    continue
                return title
        except Exception:
            _log.warning("[节日源] 日历查询异常，本次回落内置表", exc_info=True)
        return None
