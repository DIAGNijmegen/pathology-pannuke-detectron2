import argparse
from pathlib import Path
from typing import Optional, Tuple

from evalutils import DetectionEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator


class PannukeDetectionEvaluation(DetectionEvaluation):
    def __init__(
        self,
        ground_truth_path: Path,
        predictions_path: Path,
        output_file: Path,
        detection_threshold: float = 0.5,
        labels: Tuple = (),
    ):
        super().__init__(
            ground_truth_path=ground_truth_path,
            predictions_path=predictions_path,
            output_file=output_file,
            file_loader=CSVLoader(),
            validators=(
                ExpectedColumnNamesValidator(
                    expected=("image_id", "x", "y", "confidence", "label")
                ),
            ),
            join_key="image_id",
            detection_radius=12,
            detection_threshold=detection_threshold,
        )
        self._labels = labels

    def get_points(self, *, case, key):
        """
        Converts the set of ground truth or predictions for this case, into
        points that represent true positives or predictions
        """
        try:
            points = case.loc[key]
        except KeyError:
            # There are no ground truth/prediction points for this case
            return []

        return [
            (p["x"], p["y"])
            for _, p in points.iterrows()
            if p["confidence"] > self._detection_threshold and p["label"] in self._labels
        ]


def main():
    # create argument parser
    argument_parser = argparse.ArgumentParser(description="Pannuke Detection Evaluator")
    argument_parser.add_argument("--ground_truth_path", required=True)
    argument_parser.add_argument("--predictions_path", required=True)
    argument_parser.add_argument("--output_file", required=True)
    argument_parser.add_argument("--threshold", required=True)
    argument_parser.add_argument("--labels", nargs="+", required=True)

    args = vars(argument_parser.parse_args())
    PannukeDetectionEvaluation(
        Path(args["ground_truth_path"]),
        Path(args["predictions_path"]),
        Path(args["output_file"]),
        float(args["threshold"]),
        tuple(args["labels"]),
    ).evaluate()


if __name__ == "__main__":
    main()
