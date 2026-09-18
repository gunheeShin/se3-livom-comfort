"""Incremental IMU/LiDAR synchronizer for the live ROS nodes.

The offline path (``datasets/rosbag.py::synchronize``) receives every IMU and
scan up front and partitions them in one pass. A live node instead gets them
one message at a time from ROS callbacks, so this class buffers what has arrived
and emits a frame as soon as its LiDAR scan is fully covered by IMU.

The emit rule is identical to the batch pass, so an online run yields exactly
the same frames as the offline run (see ``tests/test_online_sync.py``): the batch
version's permanent ``break`` (last IMU does not yet cover the scan end) becomes
"wait for more IMU" here -- the scan stays buffered until a later ``drain`` sees
enough IMU.
"""

import numpy as np
from collections import deque


class OnlineSynchronizer:
    """Buffers incoming IMU rows and LiDAR scans, draining synced frames.

    ``drain`` returns ``(scan, imu_block)`` tuples -- the same shape the batch
    ``synchronize`` yields -- with IMU partitioned non-overlapping across scans.
    """

    def __init__(self):
        self._imus = []          # rows [t, ax, ay, az, gx, gy, gz], time-ordered
        self._imu_idx = 0        # first not-yet-consumed IMU (monotonic)
        self._scans = deque()    # pending scans in arrival (time) order

    def add_imu(self, row):
        self._imus.append(np.asarray(row, dtype=np.float64))

    def add_scan(self, header_ts, pts, offsets):
        self._scans.append(
            {
                "header_ts": float(header_ts),
                "pts": pts,
                "offsets": np.asarray(offsets, dtype=np.float64),
            }
        )

    def drain(self):
        """Emit every frame that can be synced with the IMU seen so far."""
        out = []
        while self._scans:
            scan = self._scans[0]
            if scan["offsets"].size == 0:
                self._scans.popleft()
                continue  # empty scan -> skip (matches batch)
            lidar_end = scan["header_ts"] + scan["offsets"][-1]
            # Need IMU coverage past the scan end. Not there yet -> wait; the
            # scan stays at the front for a later drain (batch's `break`).
            if not self._imus or self._imus[-1][0] < lidar_end:
                break
            # No IMU strictly before the scan end -> the node drops this scan.
            if self._imu_idx >= len(self._imus) or self._imus[self._imu_idx][0] >= lidar_end:
                self._scans.popleft()
                continue
            start = self._imu_idx
            while self._imu_idx < len(self._imus) and self._imus[self._imu_idx][0] < lidar_end:
                self._imu_idx += 1
            imu_block = np.array(self._imus[start:self._imu_idx], dtype=np.float64)
            self._scans.popleft()
            if imu_block.shape[0] == 0:
                continue
            out.append((scan, imu_block))
        return out
