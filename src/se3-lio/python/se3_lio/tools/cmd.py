"""`se3_lio_pipeline` — run SE(3)-LIO over a ROS2 rosbag and write a TUM trajectory."""

import argparse
from pathlib import Path

from se3_lio.config import load_node_params
from se3_lio.datasets import RosbagDataset, Ros1BagDataset
from se3_lio.pipeline import OdometryPipeline

# Rerun visualization tuning -- hardcoded, not CLI flags.
RERUN_KEYFRAME_DIST = 0.5   # log a keyframe scan every N meters of travel
RERUN_SUBMAP = 20           # sliding submap: keep only the last N keyframe scans
RERUN_AXIS_LENGTH = 1.5     # length of the sensor TF axes


def run():
    ap = argparse.ArgumentParser(
        prog="se3_lio_pipeline",
        description="Run SE(3)-LIO odometry over a rosbag (ROS2/Livox or ROS1/Ouster).",
    )
    ap.add_argument(
        "bag", nargs="+",
        help="ROS2 rosbag dir (Livox) or ROS1 .bag file(s) (Ouster/Hesai/Velodyne). "
        "Pass multiple .bag files to merge them (e.g. split LiDAR + IMU bags).",
    )
    ap.add_argument(
        "--config",
        dest="config",
        help="dataset config yaml (extrinsic; optional topics + algorithm overrides). "
        "Unset keys fall back to the SE3LIOConfig code defaults.",
    )
    ap.add_argument("--params", dest="config", help=argparse.SUPPRESS)  # deprecated alias
    ap.add_argument("--imu-topic", help="override imu_topic from config")
    ap.add_argument("--lidar-topic", help="override lidar_topic from config")
    ap.add_argument(
        "--input-type",
        choices=["auto", "ros2-livox", "ros1-ouster"],
        default="auto",
        help="input format (auto: .bag -> ros1-ouster, directory -> ros2-livox)",
    )
    ap.add_argument("--max-frames", type=int, default=None, help="limit number of frames")
    ap.add_argument(
        "--output", default="results", help="output directory for the TUM trajectory"
    )
    ap.add_argument("--no-progress", action="store_true", help="disable progress bar")
    ap.add_argument(
        "--visualize", action="store_true", help="open a live Polyscope viewer (needs se3-lio[viz])"
    )
    ap.add_argument("--rerun-save", help="write a Rerun .rrd recording to this path (needs se3-lio[rerun])")
    args = ap.parse_args()
    if not args.config:
        ap.error("--config is required")

    # One bag -> pass the path straight through; several -> a list the ROS1 reader
    # merges (split LiDAR/IMU bags, e.g. grandtour hesai + livox_imu).
    bag = args.bag[0] if len(args.bag) == 1 else args.bag
    input_type = args.input_type
    if input_type == "auto":
        input_type = ("ros1-ouster" if len(args.bag) > 1 or Path(args.bag[0]).is_file()
                      else "ros2-livox")

    params = load_node_params(args.config)
    if args.imu_topic:
        params["imu_topic"] = args.imu_topic
    if args.lidar_topic:
        params["lidar_topic"] = args.lidar_topic
    print(
        f"bag={bag}  input={input_type}\n"
        f"imu_topic={params['imu_topic']}  lidar_topic={params['lidar_topic']}  "
        f"min_range={params['min_range']}"
    )

    ds_cls = Ros1BagDataset if input_type == "ros1-ouster" else RosbagDataset
    dataset = ds_cls(bag, params["imu_topic"], params["lidar_topic"], params["min_range"], args.max_frames)

    logger = None
    if args.visualize:
        from se3_lio.viz import PolyscopeVisualizer

        logger = PolyscopeVisualizer(params["extrinsic"])
    elif args.rerun_save:
        from se3_lio.viz import RerunLogger

        logger = RerunLogger(
            params["extrinsic"],
            keyframe_dist=RERUN_KEYFRAME_DIST,
            save_path=args.rerun_save,
            window=RERUN_SUBMAP,
            axis_length=RERUN_AXIS_LENGTH,
        )

    pipeline = OdometryPipeline(dataset, params["config"], params["extrinsic"])
    pipeline.run(progress=not args.no_progress, logger=logger)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.bag[0].rstrip('/')).name}_se3lio.tum"
    pipeline.save_tum(out_path)

    print(pipeline.summary())
    print(f"trajectory -> {out_path}")
    if args.rerun_save:
        print(f"rerun -> {args.rerun_save}")

    if args.visualize and logger is not None:
        logger.hold()


if __name__ == "__main__":
    run()
