from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from workspace import store, scheduler, runs, effects, connections


class ScheduleRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = patch.object(store, "ROOT", Path(self.temp.name))
        self.root.start()
        self.token = store.USER.set("local")

    def tearDown(self):
        store.USER.reset(self.token)
        self.root.stop()
        self.temp.cleanup()

    def scheduled(self, **extra):
        return store.job_save({"title": "Briefing", "type": "agent", "prompt": "Read a note", "mode": "local", "model": "fixture:4b", "due": (datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat(), **extra})

    def test_dispatch_is_not_completion_and_failure_propagates(self):
        self.scheduled()
        with patch.object(runs, "resume") as resume:
            scheduler.tick(False)
            scheduler.tick(False)
            self.assertEqual(resume.call_count, 1)
        job = store.jobs()[0]
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["occurrences"][0]["status"], "queued")
        self.assertEqual(store.run_get(job["run_id"])["payload"]["model"], "fixture:4b")
        store.run_save(job["run_id"], status="failed")
        scheduler.tick(False)
        self.assertEqual(store.jobs()[0]["status"], "failed")
        self.assertEqual(store.jobs()[0]["occurrences"][0]["status"], "failed")

    def test_two_dispatchers_claim_one_occurrence(self):
        self.scheduled()
        errors = []
        def tick():
            try:
                scheduler.tick(False)
            except Exception as exc:
                errors.append(exc)
        with patch.object(runs, "resume") as resume:
            threads = [threading.Thread(target=tick) for _ in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(resume.call_count, 1)
        self.assertEqual(len(store.jobs()[0]["occurrences"]), 1)

    def test_queued_crash_recovers_but_inflight_is_interrupted(self):
        self.scheduled()
        with patch.object(runs, "resume"):
            scheduler.tick(False)
        key = store.jobs()[0]["run_id"]
        with patch.object(runs, "resume") as resume:
            scheduler.recover(False)
            resume.assert_called_once_with(key, False)
        store.run_save(key, status="running")
        with patch.object(runs, "resume") as resume:
            scheduler.recover(False)
            resume.assert_not_called()
        self.assertEqual(store.jobs()[0]["status"], "interrupted")
        self.assertEqual(store.run_get(key)["status"], "interrupted")

    def test_repeat_coalesces_missed_times_preserves_anchor(self):
        due = "2026-10-09T00:00:00+00:00"
        now = datetime.fromisoformat("2026-10-09T02:07:00+00:00")
        self.assertEqual(scheduler.next_due(due, {"repeat_minutes": 30}, now), "2026-10-09T02:30:00+00:00")

    def test_daily_sydney_handles_dst_gap_and_fold(self):
        p = {"recurrence": "daily", "timezone": "Australia/Sydney", "due": "2026-10-03T02:30:00+10:00"}
        due = scheduler.next_due(p["due"], p, datetime.fromisoformat("2026-10-03T04:00:00+10:00"))
        self.assertEqual(due, "2026-10-03T16:30:00+00:00")  # gap rolls to 03:30 AEDT
        p["due"] = "2026-04-04T02:30:00+11:00"
        due = scheduler.next_due(p["due"], p, datetime.fromisoformat("2026-04-04T04:00:00+11:00"))
        self.assertEqual(due, "2026-04-04T15:30:00+00:00")  # first fold only

    def test_reminder_recovery_does_not_duplicate(self):
        store.job_save({"title": "Reminder", "due": "2026-01-01T00:00:00+00:00"})
        scheduler.tick(False)
        scheduler.recover(False)
        scheduler.tick(False)
        self.assertEqual(len(store.items("notification")), 1)
        self.assertEqual(len(store.jobs()[0]["occurrences"]), 1)

    def test_stale_draft_approval_does_not_send(self):
        item = store.save({"kind": "draft", "title": "Fixture", "content": "Reviewed"})
        store.save({"content": "Changed"}, item["id"])
        with patch.object(connections, "google") as google:
            with self.assertRaisesRegex(ValueError, "changed"):
                effects.publish("mail_send", item["id"], item["updated"], lambda item: connections.mail_send(item["id"], item))
            google.assert_not_called()

    def test_uncertain_calendar_publish_cannot_repeat(self):
        item = store.save({"kind": "event", "title": "Fixture", "content": "Review", "meta": {"start": "2026-10-10T10:00:00+11:00", "end": "2026-10-10T11:00:00+11:00"}})
        with patch.object(connections, "google", side_effect=TimeoutError) as google:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "uncertain"):
                    current = store.get(item["id"])
                    effects.publish("calendar_publish", item["id"], current["updated"], lambda item: connections.calendar_publish(item["id"], item))
            self.assertEqual(google.call_count, 1)
            self.assertEqual(len(google.call_args.args[1]["id"]), 64)

    def test_reconciliation_records_user_evidence_without_replay(self):
        item = store.save({"kind": "event", "title": "Fixture", "content": "Review", "meta": {"start": "2026-10-10T10:00:00+11:00", "end": "2026-10-10T11:00:00+11:00"}})
        with patch.object(connections, "google", side_effect=TimeoutError) as google:
            with self.assertRaises(RuntimeError):
                effects.publish("calendar_publish", item["id"], item["updated"], lambda item: connections.calendar_publish(item["id"], item))
            with store.db() as conn:
                action_id = conn.execute("SELECT id FROM actions").fetchone()[0]
            result = effects.reconcile(action_id, "confirmed_not_applied", "Checked the destination calendar: no event exists")
            self.assertEqual(google.call_count, 1)
            self.assertEqual(result["source"], "user_attestation")
            self.assertFalse(store.get(item["id"])["meta"]["publishing"])

    def test_scheduler_has_one_owner_for_workspace(self):
        first = scheduler.Scheduler(False)
        try:
            with self.assertRaisesRegex(RuntimeError, "Another Nana scheduler"):
                scheduler.Scheduler(False)
        finally:
            first.stop.set()
            first.thread.join(2)


if __name__ == "__main__":
    unittest.main()
