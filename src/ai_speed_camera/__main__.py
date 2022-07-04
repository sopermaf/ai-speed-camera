"""Command-line interface."""
import click
import csv
import json
import logging

from .annotations_processor import extract_cars
from .video import annotate_frames

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--video", help="the video file to process", type=click.File("r"))
@click.option("--output", help="the processed output video file")
@click.option("--annotations",help="annotation response JSON file from Google Cloud Video API",required=True,type=click.File("r"))
@click.option("--distance",help="horizontal distance (metres) captured by video",required=True,type=int,)
@click.option("--frame-rate", help="frame rate for video file", type=int, default=15)
@click.option("--width", help="width for input video file", type=int, default=1920)
@click.option("--height", help="heightfor input video file", type=int, default=1080)
@click.option("--min-speed",help="ignore cars travelling slower than threshold (kmph)",type=int,default=1,)
@click.option("--min-distance",help="ignore cars travelling  minimum relative distance across frame (0..1)",type=float,default=0,)
@click.option("--export-to-csv", is_flag=True, default=False, help="export detected cars details to csv file")
def main(video, output, annotations, distance, frame_rate, width, height, min_speed, min_distance, export_to_csv) -> None:
    """Ai Speed Camera."""
    print(video, output)
    results = json.loads(annotations.read())
    cars_frame_lookup = extract_cars(
        results, frame_rate, distance, min_speed, min_distance
    )

    if export_to_csv:
        logging.info("Exporting valid car statistics to csv file")
        with open(export_to_csv, mode="w", newline="") as csv_file:
            to_csv = csv.writer(csv_file)

            to_csv.writerow(
                ["Car Detected #", "Entrance Time (s)", "Exit Time (s)", "Speed (km/h)"]
            )
            for idx, car in enumerate(cars_frame_lookup):
                to_csv.writerow(
                    [idx, car["entrance_time"], car["exit_time"], car["car_speed"]]
                )

    if video is not None and output is not None:
        logging.info("Processing source video file")
        annotate_frames(
            cars_frame_lookup,
            video.name,
            output,
            frame_rate,
            width,
            height,
        )
    else:
        logging.info(
            "Missing video and output parameters - skipping source video annotation."
        )

if __name__ == "__main__":
    main(prog_name="ai-speed-camera")  # pragma: no cover