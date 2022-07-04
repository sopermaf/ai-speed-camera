import functools
import itertools
import logging
import math
import re

import numpy


seconds = re.compile(r"\d+(\.\d+)?")


def _line_midpoint(start, end):
    return (start + end) / 2


def _bb_centroid(bb):
    return (
        _line_midpoint(bb["left"], bb["right"]),
        _line_midpoint(bb["top"], bb["bottom"]),
    )


def _abs_distance(x1, x2):
    return math.sqrt((x2 - x1) ** 2)


def _mps_to_khm(mps):
    return round((mps * 3.6), 2)


def _time_to_seconds(time):
    return seconds.match(time)


def _seconds_to_frame_number(seconds, frame_rate):
    return int(seconds * frame_rate)


def _frame_number_to_seconds(frame_number, frame_rate):
    return frame_number / frame_rate


def _is_car(oa):
    return oa["entity"]["description"] == "car"


default_bb = {"left": 0, "right": 0, "top": 0, "bottom": 0}


def _frame_to_box_lookup(frame, frame_rate):
    m = _time_to_seconds(frame["timeOffset"])
    if m is not None:
        seconds = float(m.group())
        frame_number = _seconds_to_frame_number(seconds, frame_rate)
        bounding_box = default_bb | frame["normalizedBoundingBox"]
        return (frame_number, bounding_box)
    else:
        return (None, None)


def _pairwise(iterable):
    """Yields tuples of the iterable in pairs.

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _merge_lookups(lookup, frame_box):
    (frame_number, bounding_box) = frame_box
    lookup[frame_number] = bounding_box
    return lookup


def _generate_bounding_boxes(start_bb, end_bb, steps):
    """Boxes generated between positions using number of intermediate steps."""

    def step_coords(coord):
        return numpy.linspace(
            start_bb[coord], end_bb[coord], num=steps, endpoint=False
        )[1:]

    bb_coords = map(step_coords, ["left", "top", "right", "bottom"])
    bounding_boxes = map(lambda coords: _to_bb(*coords), zip(*bb_coords))
    return bounding_boxes


def _to_bb(left, top, right, bottom):
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


# Annotations frame rate may differ from frame rate of the video source.
# Estimate bounding boxes in missing frame annotaitons by using linear
# interpolation between known bounding box frames.
def _add_missing_frames(frames):
    frame_pairs = _pairwise(frames)
    all_frames = []
    for (start, end) in frame_pairs:
        all_frames.append(start)
        start_index = start[0]
        end_index = end[0]
        start_bb = start[1]
        end_bb = end[1]
        steps = end_index - start_index
        missing_bbs = _generate_bounding_boxes(start_bb, end_bb, steps)
        missing_frames = [
            (index, bb) for index, bb in enumerate(missing_bbs, start=(start_index + 1))
        ]
        all_frames.extend(missing_frames)
        all_frames.append(end)
    return all_frames


# Calculate relative distance travelled across X axis in frame (0..1)
# between centres of two relative bounding boxes within frame
def _relative_distance_traveled(bb_start, bb_end):
    start_centroid = _bb_centroid(bb_start)
    end_centroid = _bb_centroid(bb_end)
    return _abs_distance(start_centroid[0], end_centroid[0])


def _calculate_speed(frames, frame_rate, distance, index, start_time, end_time):
    """Calculate speed of car from frames.

    - Use start and end bounding box locations as start and end positions.
    - Use centroid of the bounding box as estimated car position.
    - Bounding boxes are normalised within frame (0 -> 1) need to convert
        relative distance travelled across the frame back to known distance
        covered by the frame in the video.
    """
    start, end = frames[0], frames[-1]
    rdt = _relative_distance_traveled(start[1], end[1])
    actual_distance_traveled = distance * rdt
    time = _frame_number_to_seconds(end[0], frame_rate) - _frame_number_to_seconds(
        start[0], frame_rate
    )
    speed = _mps_to_khm(actual_distance_traveled / time) if time > 0 else 0.0
    logging.debug(
        (
            "car #%s: speed = %6.2f\t(start = %s,  end = %s,  "
            "distance = %6.2f,  time = %.2f)"
        ),
        index,
        speed,
        start_time,
        end_time,
        actual_distance_traveled,
        time,
    )
    return speed


def _parse_annotation(index, annotation, frame_rate, distance):
    frames = annotation["frames"]
    frame_boxes = list(map(lambda f: _frame_to_box_lookup(f, frame_rate), frames))
    all_frame_boxes = _add_missing_frames(frame_boxes)
    frame_box_lookup = functools.reduce(_merge_lookups, all_frame_boxes, {})
    frame_box_lookup["entrance_time"] = frames[0]["timeOffset"]
    frame_box_lookup["exit_time"] = frames[-1]["timeOffset"]
    frame_box_lookup["car_speed"] = _calculate_speed(
        frame_boxes,
        frame_rate,
        distance,
        index,
        frame_box_lookup["entrance_time"],
        frame_box_lookup["exit_time"],
    )
    frame_box_lookup["rdt"] = _relative_distance_traveled(
        frame_boxes[0][1], frame_boxes[-1][1]
    )
    frame_box_lookup["index"] = index
    return frame_box_lookup


def _is_car_valid(car, min_speed, min_rdt):
    return car["car_speed"] >= min_speed and car["rdt"] >= min_rdt


def _extract_cars(annotations_response, frame_rate, distance, min_speed, min_rdt):
    ar = annotations_response["response"]["annotationResults"][0]["objectAnnotations"]
    cars = list(filter(_is_car, ar))
    logging.info("Discovered " + str(len(cars)) + " total cars in annotation response")
    cars_frame_boxes_lookup = list(
        map(
            lambda args: _parse_annotation(args[0], args[1], frame_rate, distance),
            enumerate(cars),
        )
    )
    valid_cars = list(
        filter(
            lambda car: _is_car_valid(car, min_speed, min_rdt), cars_frame_boxes_lookup
        )
    )
    logging.info(
        "Discovered "
        + str(len(valid_cars))
        + " valid cars in annotation response: "
        + ", ".join(list(map(lambda c: str(c["index"]), valid_cars)))
    )
    return valid_cars
