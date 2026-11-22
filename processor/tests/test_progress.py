"""Keep periodic phone meters alive independently of slow diagnostic summaries."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import os
os.environ.setdefault("LIPSYNC_LOG_FILE", "off")  # keep the session log clean
from processor import server


class ProgressTests(unittest.TestCase):
    def test_phone_and_viewer_updates_continue_between_five_second_logs(self):
        stats = server.ArrivalStats()
        stats.blocks = 250
        chain = SimpleNamespace(in_db=-30, out_db=-31)
        with patch.object(server, 'arrivals', stats), patch.object(server.hub, 'publish') as publish, \
                patch.object(server.test_mode, 'tick') as tick, \
                patch.object(server.test_mode, 'summary', return_value='summary') as summary, \
                patch.object(server.debug, 'log') as log:
            for seq in (0, 5, 25, 249):
                server.report_audio_progress(seq, 'test-session', chain)
            self.assertEqual(tick.call_count, 3)
            self.assertEqual(publish.call_count, 3)
            self.assertEqual([c.kwargs['include_phone'] for c in publish.call_args_list], [True, False, True])
            self.assertTrue(all(c.args[0]['session_id'] == 'test-session' for c in publish.call_args_list))
            summary.assert_called_once()
            self.assertEqual(log.call_count, 2)
            self.assertIn('250 blocks', log.call_args_list[0].args[2])
            self.assertEqual(stats.blocks, 0)

    def test_network_reset_keeps_the_next_interarrival_gap_measurable(self):
        stats = server.ArrivalStats()
        with patch.object(server.time, 'perf_counter', side_effect=[1, 1.02, 1.22]):
            stats.arrived()
            stats.arrived()
            stats.reset()
            stats.arrived()
        self.assertEqual(stats.status(), {'blocks': 1, 'max_gap_ms': 200, 'gaps_over_100ms': 1})


if __name__ == '__main__':
    unittest.main()
