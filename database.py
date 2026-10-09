import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path('results/parking.db')
HOURLY_RATE = 5000

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS parking_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            time_in TEXT NOT NULL,
            time_out TEXT,
            fee INTEGER NOT NULL DEFAULT 0,
            image_name TEXT,
            detection_confidence REAL
        )''')
        con.execute('CREATE INDEX IF NOT EXISTS idx_sessions_plate ON parking_sessions(plate_number)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_sessions_exit ON parking_sessions(time_out)')

def enter(plate, image_name=None, confidence=None):
    plate = plate.strip().upper()
    if not plate:
        return False, 'Biển số không được để trống.'
    with connect() as con:
        existing = con.execute('SELECT id FROM parking_sessions WHERE plate_number=? AND time_out IS NULL', (plate,)).fetchone()
        if existing:
            return False, f'Xe {plate} đang ở trong bãi (lượt #{existing["id"]}).'
        con.execute('INSERT INTO parking_sessions(plate_number,time_in,image_name,detection_confidence) VALUES(?,?,?,?)',
                    (plate, datetime.now().isoformat(timespec='seconds'), image_name, confidence))
    return True, f'Đã ghi nhận xe {plate} vào bãi.'

def exit_vehicle(plate):
    plate = plate.strip().upper()
    with connect() as con:
        row = con.execute('SELECT * FROM parking_sessions WHERE plate_number=? AND time_out IS NULL ORDER BY id DESC LIMIT 1', (plate,)).fetchone()
        if not row:
            return False, f'Không tìm thấy xe {plate} đang ở trong bãi.', 0
        now = datetime.now()
        seconds = max(0, (now - datetime.fromisoformat(row['time_in'])).total_seconds())
        # Tính tối thiểu 1 giờ; làm tròn lên theo từng giờ
        hours = max(1, int((seconds + 3599) // 3600))
        fee = hours * HOURLY_RATE
        con.execute('UPDATE parking_sessions SET time_out=?, fee=? WHERE id=?',
                    (now.isoformat(timespec='seconds'), fee, row['id']))
    return True, f'Xe {plate} đã ra bãi. Thời gian tính phí: {hours} giờ.', fee

def sessions(limit=1000):
    with connect() as con:
        return [dict(row) for row in con.execute('SELECT * FROM parking_sessions ORDER BY id DESC LIMIT ?', (limit,))]

def stats():
    with connect() as con:
        row = con.execute('''SELECT COUNT(*) AS total,
            SUM(CASE WHEN time_out IS NULL THEN 1 ELSE 0 END) AS active,
            COALESCE(SUM(fee),0) AS revenue,
            SUM(CASE WHEN time_out IS NOT NULL THEN 1 ELSE 0 END) AS completed
            FROM parking_sessions''').fetchone()
        return {k: row[k] or 0 for k in row.keys()}
