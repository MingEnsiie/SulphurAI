import io
import unittest

from src.progress import ProgressReporter, format_duration


class ProgressReporterTest(unittest.TestCase):
    def test_format_duration_uses_seconds_minutes_and_hours(self):
        self.assertEqual(format_duration(5), "5s")
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_duration(3661), "1h 01m")

    def test_step_callback_prints_percentage_and_eta(self):
        ticks = iter([100.0, 110.0])
        stream = io.StringIO()
        reporter = ProgressReporter("Sampling", total=4, stream=stream, time_fn=lambda: next(ticks))

        result = reporter.step_callback(None, 0, 123, {"latents": object()})

        self.assertEqual(result, {})
        output = stream.getvalue()
        self.assertIn("Sampling", output)
        self.assertIn("1/4", output)
        self.assertIn("25.0%", output)
        self.assertIn("ETA 30s", output)

    def test_finish_prints_newline(self):
        stream = io.StringIO()
        reporter = ProgressReporter("Saving", total=1, stream=stream, time_fn=lambda: 1.0)

        reporter.update(1)
        reporter.finish()

        self.assertTrue(stream.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
