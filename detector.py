from pathlib import Path

from ultralytics import YOLO


MODEL_CANDIDATES = [
    Path("models/new_model/best.pt"),
    Path("models/license_plate_new.pt"),
    Path("models/license_plate.pt"),
    Path("models/best.pt"),
]


class LicensePlateDetector:
    def __init__(self, model_path=None):
        path = (
            Path(model_path)
            if model_path
            else next((p for p in MODEL_CANDIDATES if p.is_file()), None)
        )

        if path is None or not path.is_file():
            raise FileNotFoundError(
                "Không tìm thấy model YOLO trong thư mục models/."
            )

        self.model_path = str(path)
        self.model = YOLO(self.model_path)

        names = self.model.names
        labels = list(names.values()) if isinstance(names, dict) else list(names)

        if len(labels) > 10 or not any(
            "plate" in str(label).lower()
            or "bienso" in str(label).lower()
            for label in labels
        ):
            raise ValueError(
                f"Model không phù hợp để nhận diện biển số: {labels}"
            )

    def detect(self, image, confidence=0.30, imgsz=960, retry=True):
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        def infer(threshold, size, source_name):
            results = self.model.predict(
                source=image,
                conf=threshold,
                iou=0.45,
                imgsz=size,
                max_det=30,
                verbose=False,
            )

            found = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(
                    lambda v: int(round(v)),
                    box.xyxy[0].tolist(),
                )

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                bw = x2 - x1
                bh = y2 - y1

                if bw < 12 or bh < 8:
                    continue

                area_ratio = (bw * bh) / (w * h)
                aspect_ratio = bw / bh

                if not (
                    0.00012 <= area_ratio <= 0.28
                    and 0.55 <= aspect_ratio <= 6.8
                ):
                    continue

                pad_x = int(bw * 0.05)
                pad_top = int(bh * 0.20)
                pad_bottom = int(bh * 0.05)

                bbox = (
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_top),
                    min(w, x2 + pad_x),
                    min(h, y2 + pad_bottom),
                )

                score = float(box.conf[0])

                found.append({
                    "bbox": bbox,
                    "confidence": score,
                    "source": source_name,
                    "needs_review": score < 0.30,
                })

            return found

        # Lần 1: nhận diện thông thường
        found = infer(confidence, imgsz, "YOLO")

        # Lần 2: chỉ chạy nếu lần đầu không tìm thấy
        if not found and retry:
            found = infer(
                threshold=0.10,
                size=960,
                source_name="YOLO (low-confidence retry)",
            )

        found.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        kept = []

        for item in found:
            x1, y1, x2, y2 = item["bbox"]
            duplicate = False

            for other in kept:
                a, b, c, d = other["bbox"]

                intersection = (
                    max(0, min(x2, c) - max(x1, a))
                    * max(0, min(y2, d) - max(y1, b))
                )

                union = (
                    (x2 - x1) * (y2 - y1)
                    + (c - a) * (d - b)
                    - intersection
                )

                if union > 0 and intersection / union > 0.45:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(item)

        return kept

    def crop(self, image, bbox, padding=0.02):
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        bw = x2 - x1
        bh = y2 - y1

        px = int(bw * padding)
        py = int(bh * padding)

        return image[
            max(0, y1 - py):min(h, y2 + py),
            max(0, x1 - px):min(w, x2 + px),
        ]