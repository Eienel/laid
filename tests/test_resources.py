import tempfile
import unittest
from pathlib import Path

from bakeoff.resources import _linux_memory


class LinuxMemoryTests(unittest.TestCase):
    def test_reads_proc_meminfo_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text(
                "MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n",
                encoding="utf-8",
            )

            total, available = _linux_memory(meminfo)

        self.assertEqual(total, 16384000 * 1024)
        self.assertEqual(available, 8192000 * 1024)

    def test_rejects_incomplete_meminfo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text("MemTotal: 1024 kB\n", encoding="utf-8")

            with self.assertRaisesRegex(OSError, "MemAvailable"):
                _linux_memory(meminfo)


if __name__ == "__main__":
    unittest.main()
