import os
from pathlib import Path
import cv2
import pandas as pd
from detector import LicensePlateDetector
from ocr import LicensePlateOCR

# Use tqdm if available, otherwise just use a simple counter
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def evaluate_pipeline(dataset_dir="datasets/biensoxemayhon100bien"):
    img_dir = Path(dataset_dir) / "anh"
    lbl_dir = Path(dataset_dir) / "label"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Không tìm thấy thư mục ảnh hoặc nhãn trong {dataset_dir}")
        return

    print("Đang khởi tạo YOLO và EasyOCR...")
    detector = LicensePlateDetector()
    ocr = LicensePlateOCR()
    
    results = []
    correct_count = 0
    missed_count = 0
    incorrect_count = 0
    
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Bắt đầu kiểm thử trên {len(image_files)} ảnh...")
    
    for img_name in tqdm(image_files):
        img_path = img_dir / img_name
        lbl_name = img_path.stem + ".txt"
        lbl_path = lbl_dir / lbl_name
        
        # Đọc nhãn gốc (Ground Truth)
        if not lbl_path.exists():
            continue
            
        with open(lbl_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip().replace("-", "").replace(".", "").replace(" ", "").upper()
            
        # Đọc ảnh
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        # Phát hiện biển số (YOLO)
        predictions = detector.detect(image, confidence=0.3)
        if not predictions:
            missed_count += 1
            results.append({"image": img_name, "truth": gt_text, "predict": "", "status": "Missed (YOLO)"})
            continue
            
        # Ưu tiên khung hình có độ tin cậy cao nhất
        pred = predictions[0]
        crop = detector.crop(image, pred['bbox'])
        
        # Nhận diện chữ (OCR)
        ocr_result = ocr.recognize(crop)
        predicted_text = ocr_result.get('text', '')
        
        if predicted_text == gt_text:
            correct_count += 1
            status = "Correct"
        else:
            incorrect_count += 1
            status = "Incorrect"
            
        results.append({
            "image": img_name, 
            "truth": gt_text, 
            "predict": predicted_text, 
            "raw": ocr_result.get('raw', ''),
            "status": status,
            "yolo_conf": pred['confidence'],
            "ocr_conf": ocr_result.get('score', 0)
        })

    # Tổng kết
    total = correct_count + incorrect_count + missed_count
    print("\n" + "="*40)
    print(" BÁO CÁO KẾT QUẢ KIỂM THỬ (TEST REPORT)")
    print("="*40)
    print(f"Tổng số ảnh: {total}")
    if total > 0:
        print(f"Đúng (Correct): {correct_count} ({correct_count/total*100:.1f}%)")
        print(f"Sai OCR (Incorrect): {incorrect_count} ({incorrect_count/total*100:.1f}%)")
        print(f"YOLO bỏ sót (Missed): {missed_count} ({missed_count/total*100:.1f}%)")
    
    # Lưu báo cáo chi tiết ra CSV
    df = pd.DataFrame(results)
    report_path = "test_report.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu chi tiết từng ảnh vào file: {report_path}")
    
    # In ra các ảnh bị sai để tiện phân tích
    print("\nDanh sách các ảnh bị sai OCR (Tối đa 10 ảnh đầu tiên):")
    errors = df[df['status'] == 'Incorrect'].head(10)
    for _, row in errors.iterrows():
        print(f"- {row['image']}: Thực tế [{row['truth']}] | OCR đọc [{row['predict']}] (Raw: {row['raw']})")

if __name__ == "__main__":
    evaluate_pipeline()
