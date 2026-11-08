import json
import time
from dataclasses import replace

import pytest

from src.bus import Bus
from src.contracts import RunLogRecord
from src.io.run_log import RunLog, start_run_log
from tests.unit.test_contracts import make_record

pytestmark = pytest.mark.unit

STAMP = time.struct_time((2026, 11, 7, 20, 15, 30, 5, 311, 0))


def test_every_tick_lands_as_one_json_line(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_LOG", "1")
    bus = Bus()
    log = start_run_log(bus, root=tmp_path, clock=lambda: STAMP)
    first = make_record()
    second = replace(first, t_s=13.0, armed=False)
    bus.publish("tick", first)
    bus.publish("tick", second)
    # read before close: line buffering means nothing waits in memory
    lines = (tmp_path / "data" / "runs" / "20261107-201530.jsonl").read_text().splitlines()
    log.close()
    assert [RunLogRecord.from_dict(json.loads(line)) for line in lines] == [first, second]


def test_off_unless_run_log_is_1(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_LOG", "0")
    assert start_run_log(Bus(), root=tmp_path, clock=lambda: STAMP) is None
    assert not (tmp_path / "data").exists()


class DeadFile:
    def write(self, s):
        raise OSError("disk gone")

    def close(self):
        pass


def test_disk_error_stops_the_log_not_the_loop(tmp_path, capsys):
    log = RunLog(tmp_path / "run.jsonl")
    log._f = DeadFile()
    log.write(make_record())
    log.write(make_record())
    assert capsys.readouterr().out == "run log stopped: disk gone\n"
