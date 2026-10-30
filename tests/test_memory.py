# SPDX-License-Identifier: GPL-3.0-only
import unittest
from unittest.mock import patch

from manga_colorizer.memory import _MacRusageV4,memory_snapshot,recycle_gpu_worker


class MemoryTests(unittest.TestCase):
    def test_physical_footprint_is_distinct_from_rss(self):
        def read(pid,version,buffer):
            self.assertEqual(version,4)
            info=_MacRusageV4.from_buffer(buffer)
            info.counters[6]=500*1024**2;info.counters[7]=3*1024**3;info.counters[28]=4*1024**3
            return 0
        with patch('manga_colorizer.memory.platform.system',return_value='Darwin'),patch('manga_colorizer.memory._reader',return_value=(None,read)):
            result=memory_snapshot()
        self.assertEqual(result['rss_bytes'],500*1024**2)
        self.assertEqual(result['physical_footprint_bytes'],3*1024**3)
        self.assertEqual(result['peak_physical_footprint_bytes'],4*1024**3)

    def test_memory_read_failure_is_unavailable_not_zero(self):
        with patch('manga_colorizer.memory.platform.system',return_value='Darwin'),patch('manga_colorizer.memory._reader',return_value=(None,lambda *_:1)):
            self.assertIsNone(memory_snapshot()['physical_footprint_bytes'])

    def test_gpu_recycles_on_physical_threshold_and_has_fallback(self):
        self.assertTrue(recycle_gpu_worker('coreml-gpu',{'physical_footprint_bytes':3*1024**3},3,1))
        self.assertFalse(recycle_gpu_worker('coreml-gpu',{'physical_footprint_bytes':2*1024**3},3,7))
        self.assertFalse(recycle_gpu_worker('cpu',{'physical_footprint_bytes':4*1024**3},3,20))
        self.assertTrue(recycle_gpu_worker('coreml-gpu',None,3,8))
        self.assertFalse(recycle_gpu_worker('coreml-gpu',None,3,7))


if __name__=='__main__':unittest.main()
