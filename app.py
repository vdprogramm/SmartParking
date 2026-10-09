import hashlib
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from detector import LicensePlateDetector
from ocr import LicensePlateOCR
from database import init_db, enter, exit_vehicle, sessions, stats, HOURLY_RATE

st.set_page_config(page_title='SmartParking ANPR', page_icon='🚗', layout='wide')
Path('uploads').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
init_db()

@st.cache_resource(show_spinner='Đang tải YOLO...')
def get_detector():
    return LicensePlateDetector()

@st.cache_resource(show_spinner='Đang tải EasyOCR...')
def get_ocr():
    return LicensePlateOCR()

@st.cache_data(show_spinner=False, max_entries=100)
def detect_cached(image_bytes, confidence, imgsz):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return get_detector().detect(img, confidence=confidence, imgsz=imgsz)

@st.cache_data(show_spinner=False, max_entries=100)
def ocr_cached(crop_bytes):
    crop = cv2.imdecode(np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR)
    return get_ocr().recognize(crop)

st.title('🚗 SmartParking ANPR')
st.caption('Nhận diện biển số • Quản lý xe vào/ra • SQLite Dashboard')
recognition_tab, parking_tab, dashboard_tab = st.tabs(['📷 Nhận diện biển số', '🅿️ Xe trong bãi', '📊 Dashboard'])

with recognition_tab:
    with st.expander('⚙️ Cài đặt nhận diện', expanded=False):
        threshold = st.slider('Ngưỡng YOLO', min_value=0.05, max_value=0.80, value=0.20, step=0.05)
        image_size = st.select_slider('Kích thước YOLO', options=[640, 960, 1280, 1920], value=640)
        st.caption('Ngưỡng thấp tăng khả năng phát hiện nhưng cũng tăng báo nhầm. Kết quả cần xác nhận trước khi lưu.')
    upload = st.file_uploader('Tải ảnh biển số xe', type=['jpg','jpeg','png'])
    if upload is not None:
        data = upload.getvalue()
        image_key = hashlib.sha256(data).hexdigest()[:16]
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error('Không đọc được ảnh. Hãy chọn JPG hoặc PNG hợp lệ.')
        else:
            try:
                detector = get_detector()
                predictions = detect_cached(data, threshold, image_size)
                if not predictions:
                    st.warning('YOLO chưa tìm thấy biển số sau khi thử lại ở độ phân giải cao. Có thể nhập thủ công ở tab Xe trong bãi.')
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Ảnh gốc — không có khung nhận diện', use_container_width=True)
                else:
                    selected = st.selectbox('Chọn biển số được phát hiện', range(len(predictions)),
                                            format_func=lambda i: f'Biển số #{i+1} – YOLO {predictions[i]["confidence"]:.1%}')
                    pred = predictions[selected]
                    x1,y1,x2,y2 = pred['bbox']
                    crop = detector.crop(image, pred['bbox'])
                    crop_bytes = cv2.imencode('.jpg', crop)[1].tobytes()
                    result = ocr_cached(crop_bytes)
                    annotated = image.copy()
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(annotated, result['text'], (x1,max(25,y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                    left, right = st.columns([2,1])
                    with left:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption='YOLO detection', use_container_width=True)
                    with right:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption='Vùng biển số', use_container_width=True)
                        st.metric('Độ tin cậy YOLO', f'{pred["confidence"]:.1%}')
                        st.caption(f'EasyOCR đọc: {result["raw"] or "Không đọc được"}')
                        st.caption(f'OCR confidence: {result["score"]:.1%} | Đồng thuận: {result.get("agreement",0):.0%}')
                        if result['needs_review'] or pred['confidence'] < 0.55:
                            st.warning('⚠️ Kết quả chưa đủ tin cậy — kiểm tra biển số thực tế trước khi lưu.')
                        with st.expander('Các kết quả OCR thử nghiệm'):
                            for candidate in result['candidates']:
                                st.write(f"{candidate['variant']}: {candidate['text']} (OCR {candidate['score']:.0%})")
                        st.caption('OCR có thể sai. Hãy kiểm tra biển số trước khi ghi nhận.')
                    plate = st.text_input('Biển số (kiểm tra và sửa trước khi lưu)', value=result['text'], key=f'plate_{image_key}_{selected}').strip().upper()
                    confirmed = st.checkbox('Tôi đã đối chiếu biển số với ảnh thực tế', key=f'confirm_{image_key}_{selected}')
                    a,b = st.columns(2)
                    if a.button('🚘 Ghi nhận xe VÀO', type='primary', use_container_width=True):
                        if not plate or not confirmed:
                            st.error('Hãy nhập biển số và xác nhận đã đối chiếu ảnh.')
                        else:
                            name = f'{datetime.now():%Y%m%d_%H%M%S}_{image_key}.{upload.name.rsplit(".",1)[-1].lower()}'
                            (Path('uploads') / name).write_bytes(data)
                            ok, message = enter(plate, name, pred['confidence'])
                            (st.success if ok else st.warning)(message)
                    if b.button('🚗 Ghi nhận xe RA', use_container_width=True):
                        if not plate or not confirmed:
                            st.error('Hãy nhập biển số và xác nhận đã đối chiếu ảnh.')
                        else:
                            ok, message, fee = exit_vehicle(plate)
                            (st.success if ok else st.warning)(message)
                            if ok:
                                st.metric('Phí gửi xe', f'{fee:,} ₫')
            except Exception as exc:
                st.error(f'Lỗi nhận diện: {exc}')
                st.info('Kiểm tra file .pt và chạy lại bằng Python trong .venv.')

with parking_tab:
    st.subheader('Xe đang trong bãi')
    all_sessions = sessions()
    active = [s for s in all_sessions if s['time_out'] is None]
    if active:
        st.dataframe(pd.DataFrame(active)[['id','plate_number','time_in','image_name']],
                     use_container_width=True, hide_index=True)
    else:
        st.info('Hiện chưa có xe trong bãi.')
    st.subheader('Tra cứu / xử lý thủ công')
    manual_plate = st.text_input('Nhập biển số', key='manual_plate').strip().upper()
    c1,c2 = st.columns(2)
    if c1.button('Cho xe vào (thủ công)'):
        ok, msg = enter(manual_plate)
        (st.success if ok else st.warning)(msg)
    if c2.button('Cho xe ra (thủ công)'):
        ok, msg, fee = exit_vehicle(manual_plate)
        (st.success if ok else st.warning)(msg + (f' Phí: {fee:,} ₫' if ok else ''))
    if st.button('🔄 Làm mới danh sách'):
        st.rerun()

with dashboard_tab:
    values = stats()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric('Tổng lượt xe', values['total'])
    m2.metric('Xe đang gửi', values['active'])
    m3.metric('Xe đã ra', values['completed'])
    m4.metric('Tổng doanh thu', f'{values["revenue"]:,} ₫')
    st.caption(f'Biểu phí demo: {HOURLY_RATE:,} ₫/giờ, làm tròn lên; tối thiểu 1 giờ.')
    history = sessions()
    if history:
        df = pd.DataFrame(history)
        df['day'] = pd.to_datetime(df['time_in']).dt.strftime('%Y-%m-%d')
        left, right = st.columns(2)
        with left:
            st.subheader('Lượt xe theo ngày')
            st.bar_chart(df.groupby('day').size().rename('Lượt xe'))
        with right:
            st.subheader('Doanh thu theo ngày ra')
            exited = df[df['time_out'].notna()].copy()
            if not exited.empty:
                exited['exit_day'] = pd.to_datetime(exited['time_out']).dt.strftime('%Y-%m-%d')
                st.bar_chart(exited.groupby('exit_day')['fee'].sum())
            else:
                st.info('Chưa có xe ra bãi để thống kê doanh thu.')
        st.subheader('Lịch sử gửi xe')
        st.dataframe(df[['id','plate_number','time_in','time_out','fee','detection_confidence']],
                     use_container_width=True, hide_index=True)
        st.download_button('⬇️ Xuất CSV', data=df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='parking_history.csv', mime='text/csv')
    else:
        st.info('Chưa có dữ liệu. Hãy ghi nhận xe vào để xem thống kê.')
