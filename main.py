"""
Intelligent Traffic Violation Detection System
===============================================

Main entry point for the application.

Usage:
    # Process a video file with display window
    python main.py --video data/sample_videos/traffic.mp4 --show

    # Process with custom camera ID
    python main.py --video traffic.mp4 --camera-id CAM_002 --show

    # Start the REST API server
    python main.py --api

    # Process video AND start API (API serves results)
    python main.py --video traffic.mp4 --api
"""

import argparse
import logging
import sys
import os
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.database import create_tables, add_violation
from utils.config import API_HOST, API_PORT, DEFAULT_CAMERA_ID, DEFAULT_LOCATION


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def violation_callback(violation: dict):
    """Callback invoked when a violation is detected — saves to database."""
    try:
        add_violation(violation)
        logging.getLogger("main").info(
            f"Saved violation: {violation.get('type')} - Vehicle #{violation.get('track_id')}"
        )
    except Exception as e:
        logging.getLogger("main").error(f"Failed to save violation: {e}")


def run_video_pipeline(args):
    """Run the video processing pipeline."""
    from pipeline.video_processor import VideoProcessor, MultiCameraProcessor

    if args.video:
        processor = VideoProcessor(
            video_source=args.video,
            camera_id=args.camera_id,
            location=args.location,
            frame_skip=args.frame_skip,
            show=args.show,
            on_violation=violation_callback,
        )
        processor.run()
    else:
        logging.getLogger("main").error("No video source specified. Use --video.")


def run_api_server():
    """Launch the FastAPI server."""
    import uvicorn
    from backend.api import app

    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Traffic Violation Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video traffic.mp4 --show
  python main.py --video traffic.mp4 --camera-id CAM_002 --frame-skip 3
  python main.py --api
  python main.py --video traffic.mp4 --api --show
        """,
    )

    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video file or RTSP stream URL.",
    )
    parser.add_argument(
        "--camera-id", type=str, default=DEFAULT_CAMERA_ID,
        help=f"Camera identifier (default: {DEFAULT_CAMERA_ID}).",
    )
    parser.add_argument(
        "--location", type=str, default=DEFAULT_LOCATION,
        help=f"Location name (default: {DEFAULT_LOCATION}).",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=2,
        help="Process every Nth frame for performance (default: 2).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display annotated video in a window.",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Start the FastAPI REST server.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("main")

    # Initialize database
    create_tables()
    logger.info("Database initialized.")

    if not args.video and not args.api:
        parser.print_help()
        print("\nError: Please specify --video and/or --api.")
        sys.exit(1)

    # Start API in background thread if both modes requested
    if args.api and args.video:
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        logger.info(f"API server started at http://{API_HOST}:{API_PORT}")
        run_video_pipeline(args)
    elif args.api:
        logger.info(f"Starting API server at http://{API_HOST}:{API_PORT}")
        run_api_server()
    else:
        run_video_pipeline(args)


if __name__ == "__main__":
    main()
