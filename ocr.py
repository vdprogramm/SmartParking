import re
from collections import Counter
import cv2
import easyocr
import numpy as np

ALLOWED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


class LicensePlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    @staticmethod
    def normalize(text):
        return re.sub(r'[^A-Z0-9]', '', str(text).upper())

    @staticmethod
    def correct_characters(text):
        if len(text) < 7:
            return text

        char_to_num = {'O': '0', 'Q': '0', 'D': '0', 'U': '0', 'V': '0', 'I': '1', 'Z': '2', 'B': '8', 'S': '5', 'G': '6'}
        num_to_char = {'0': 'D', '1': 'I', '2': 'Z', '8': 'B', '5': 'S'}
        
        res = ""

        res += char_to_num.get(text[0], text[0])
        res += char_to_num.get(text[1], text[1])
        
        tail_len = 5 if len(text) >= 8 else 4
        middle = text[2:-tail_len]
        tail = text[-tail_len:]
        
        if len(middle) > 0:
            res += num_to_char.get(middle[0], middle[0])
            for i in range(1, len(middle)):
                res += middle[i]
                
        for c in tail:
            res += char_to_num.get(c, c)
            
        return res

    @staticmethod
    def plate_shape(text):
        if re.fullmatch(r'\d{2}[A-Z]\d{1,2}\d{4,5}', text):
            return 1.0
        if re.fullmatch(r'\d{2}[A-Z0-9]{1,3}\d{4,5}', text):
            return 0.55
        return 0.0

    @staticmethod
    def _variants(gray):
        h, w = gray.shape[:2]
        scale = min(6.0, max(2.0, 250 / max(1, h)))
        large = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(large, -1, kernel)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(sharpened)
        denoised = cv2.bilateralFilter(clahe, 5, 35, 35)
        binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return [('gray', large), ('sharp', sharpened), ('clahe', clahe), ('binary', binary)]

    def _read(self, img):
        try:
            pad = int(img.shape[0] * 0.15)
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            
            boxes = self.reader.readtext(
                padded, detail=1, paragraph=False, allowlist=ALLOWED,
                decoder='greedy', min_size=5, text_threshold=0.35,
                low_text=0.25, link_threshold=0.3, mag_ratio=1.5
            )
        except (ValueError, RuntimeError):
            return '', 0.0
        tokens = []
        for coords, raw, confidence in boxes:
            clean = self.normalize(raw)
            if not clean:
                continue
            x = float(np.mean([p[0] for p in coords])) - pad
            y = float(np.mean([p[1] for p in coords])) - pad
            tokens.append((y, x, clean, float(confidence)))
        if not tokens:
            return '', 0.0

        tokens.sort(key=lambda item: item[0])
        rows = []
        tolerance = max(10, img.shape[0] * 0.13)
        for token in tokens:
            group = next((row for row in rows if abs(token[0] - np.mean([t[0] for t in row])) <= tolerance), None)
            if group is None:
                rows.append([token])
            else:
                group.append(token)
        rows.sort(key=lambda row: np.mean([t[0] for t in row]))
        text = ''.join(''.join(t[2] for t in sorted(row, key=lambda t: t[1])) for row in rows)
        confidence = float(np.mean([t[3] for t in tokens]))
        return text, confidence

    def recognize(self, image):
        empty = {'text': '', 'raw': '', 'score': 0.0, 'plausibility': 0.0,
                 'agreement': 0.0, 'candidates': [], 'needs_review': True}
        if image is None or image.size == 0:
            return empty
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        h, w = gray.shape[:2]
        if h < 15 or w < 20:
            return empty

        margin_x, margin_y = int(w * 0.02), int(h * 0.02)
        if margin_x and margin_y and h - 2 * margin_y > 12 and w - 2 * margin_x > 18:
            gray = gray[margin_y:h-margin_y, margin_x:w-margin_x]
        h = gray.shape[0]
        candidates = []
        for variant_name, variant in self._variants(gray):
            text, conf = self._read(variant)
            if text:
                original_text = text
                text = self.correct_characters(text)
                candidates.append({'text': text, 'raw': original_text, 'score': conf,
                                   'plausibility': self.plate_shape(text), 'variant': f'{variant_name}:full'})

            if 0.85 <= gray.shape[1] / max(1, h) <= 3.5:
                for split in (0.46, 0.52, 0.58):
                    pivot = int(variant.shape[0] * split)
                    overlap = max(1, int(variant.shape[0] * 0.035))
                    top = variant[:min(variant.shape[0], pivot + overlap), :]
                    bottom = variant[max(0, pivot - overlap):, :]
                    top_text, top_conf = self._read(top)
                    bottom_text, bottom_conf = self._read(bottom)
                    if not top_text or not bottom_text:
                        continue
                    combined = top_text + bottom_text
                    combined = self.correct_characters(combined)
                    candidates.append({'text': combined, 'raw': f'{top_text} / {bottom_text}',
                                       'score': min(top_conf, bottom_conf),
                                       'plausibility': self.plate_shape(combined),
                                       'variant': f'{variant_name}:split{split:.2f}'})
        if not candidates:
            return empty
        votes = Counter(item['text'] for item in candidates)
        candidates.sort(key=lambda item: (item['plausibility'], votes[item['text']], item['score']), reverse=True)
        best = candidates[0]
        agreement = votes[best['text']] / len(candidates)
        return {**best, 'agreement': agreement, 'candidates': candidates,
                'needs_review': best['plausibility'] < 1.0 or best['score'] < 0.70 or agreement < 0.45}
