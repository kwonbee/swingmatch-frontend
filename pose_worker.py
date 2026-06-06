#!/usr/bin/env python3
"""
MySwingMatch — 포즈 추정 워커 (Railway 별도 파이썬 서비스, 상시 실행)
=====================================================================
설계 (노션 마스터 문서 2026-06-06 callout 확정):
  · 방식 B: 별도 파이썬 워커. golf_features_v3.py 의 v3 로직 그대로 재사용.
  · 트리거: DB 폴링. analysis_logs.pose_status='pending' 행을 주워서 처리(= DB가 큐).
  · 실행 형태: 상시 실행 + POLL_INTERVAL 초 간격 폴링 + 모델 1회 로드 후 상주.
  · 대상: 저장된 영상만. app.html 이 영상 저장 성공 시 pose_status='pending' 으로 기록.
          (비회원·무료 1·Pro 40 한도는 app.html 이 결정 → 워커는 그저 pending 만 처리)
  · 출력: pose_v3 확정본을 Storage 'swing-videos' 버킷의 pose/{logId}.json 으로 업로드.
          analysis_logs 에는 pose_status='done', pose_path 만 기록(row 경량).
  · 유저 체감 분석 시간 0 증가: 분석/응답과 완전 분리된 백그라운드 처리.

환경변수 (Railway):
  SUPABASE_URL          예: https://vldcmanngfexfwvvnqbb.supabase.co
  SUPABASE_SERVICE_KEY  service_role 키 (Storage 업로드 + 상태 UPDATE 권한 필요)
  VIDEO_BUCKET          기본 'swing-videos'
  POLL_INTERVAL         폴링 간격(초), 기본 10
  BATCH_SIZE            한 번에 가져올 pending 개수, 기본 3
  STALE_PROCESSING_MIN  processing 인 채 멈춘 행을 다시 pending 으로 되돌릴 분, 기본 15

의존성: requirements.txt 참조 (mediapipe, opencv-python-headless, numpy, requests)

로컬 테스트:
  SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python pose_worker.py
"""
import os
import io
import sys
import json
import time
import tempfile
import traceback
import urllib.request

import cv2
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://vldcmanngfexfwvvnqbb.supabase.co").rstrip("/")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
VIDEO_BUCKET = os.environ.get("VIDEO_BUCKET", "swing-videos")
POSE_BUCKET = os.environ.get("POSE_BUCKET", "pose-data")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "3"))
STALE_PROCESSING_MIN = int(os.environ.get("STALE_PROCESSING_MIN", "15"))

if not SERVICE_KEY:
    print("[FATAL] SUPABASE_SERVICE_KEY 환경변수가 없습니다.", flush=True)
    sys.exit(1)

REST = SUPABASE_URL + "/rest/v1"
STORAGE = SUPABASE_URL + "/storage/v1"
HEADERS = {
    "apikey": SERVICE_KEY,
    "Authorization": "Bearer " + SERVICE_KEY,
    "Content-Type": "application/json",
}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = os.environ.get("MODEL_PATH", "pose_landmarker_full.task")

# ─────────────────────────────────────────────────────────────────────
# v3 로직 (golf_features_v3.py 에서 가져옴 — 좌표/feature 계약 동일, 변경 금지)
# 차이점: analyze() 가 landmarker 를 인자로 받아 재사용(모델 상주). 파일 덤프 제거.
# ─────────────────────────────────────────────────────────────────────
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

SAVE_JOINTS = {
    "nose": NOSE, "l_shoulder": L_SH, "r_shoulder": R_SH,
    "l_elbow": L_EL, "r_elbow": R_EL, "l_wrist": L_WR, "r_wrist": R_WR,
    "l_hip": L_HIP, "r_hip": R_HIP, "l_knee": L_KN, "r_knee": R_KN,
    "l_ankle": L_AN, "r_ankle": R_AN,
}
SKELETON = [
    ["l_shoulder", "r_shoulder"], ["l_shoulder", "l_hip"], ["r_shoulder", "r_hip"],
    ["l_hip", "r_hip"], ["l_shoulder", "l_elbow"], ["l_elbow", "l_wrist"],
    ["r_shoulder", "r_elbow"], ["r_elbow", "r_wrist"], ["l_hip", "l_knee"],
    ["l_knee", "l_ankle"], ["r_hip", "r_knee"], ["r_knee", "r_ankle"],
]
MEASURE_LINES = {
    "spine": {"from": "_shoulder_mid", "to": "_hip_mid", "label": "척추"},
    "shoulder": {"from": "l_shoulder", "to": "r_shoulder", "label": "어깨선"},
}
VIS_SHOW = 0.5


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print("[모델] 다운로드 중...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[모델] 다운로드 완료", flush=True)


def make_landmarker():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.PoseLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    return vision.PoseLandmarker.create_from_options(opts)


def interpolate(series, vis, thresh=0.5):
    series = series.copy(); n = len(series)
    good = [i for i in range(n) if vis[i] >= thresh and not np.isnan(series[i]).any()]
    if not good:
        return series
    for i in range(n):
        if i in good:
            continue
        prev = max([g for g in good if g < i], default=None)
        nxt = min([g for g in good if g > i], default=None)
        if prev is not None and nxt is not None:
            t = (i - prev) / (nxt - prev)
            series[i] = series[prev] * (1 - t) + series[nxt] * t
        elif prev is not None:
            series[i] = series[prev]
        elif nxt is not None:
            series[i] = series[nxt]
    return series


_MP_TS = 0  # mediapipe detect_for_video 용 전역 단조증가 timestamp(ms).
            # landmarker 인스턴스를 모든 영상에 재사용하므로 영상이 바뀌어도
            # 절대 줄어들면 안 됨 → 영상 경계를 넘어 계속 증가시킨다.


def analyze(path, lm, mp):
    """영상 1개 → pose_v3 dict 1개 반환. (lm: 상주 landmarker, mp: mediapipe 모듈)
    원본 golf_features_v3.analyze 와 산출 구조 100% 동일. 파일 덤프/모델 생성만 제거."""
    global _MP_TS
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_raw, vis_raw, idx = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 전역 ts를 항상 1ms 이상 증가시켜 전달 (영상 내 겹침 + 영상 간 리셋 모두 방지)
        _MP_TS += 1
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), _MP_TS)
        if res.pose_landmarks:
            pts = np.array([[p.x, p.y] for p in res.pose_landmarks[0]])
            v = np.array([p.visibility for p in res.pose_landmarks[0]])
        else:
            pts = np.full((33, 2), np.nan); v = np.zeros(33)
        frames_raw.append(pts); vis_raw.append(v); idx += 1
    cap.release()
    n = len(frames_raw)
    if n == 0:
        return None
    frames_raw = np.array(frames_raw); vis_raw = np.array(vis_raw)
    coords = frames_raw.copy()
    for j in range(33):
        coords[:, j, :] = interpolate(coords[:, j, :], vis_raw[:, j])

    def mid(a, b): return (coords[:, a, :] + coords[:, b, :]) / 2
    sh_mid = mid(L_SH, R_SH); hip_mid = mid(L_HIP, R_HIP)
    sh_width = np.linalg.norm(coords[:, L_SH, :] - coords[:, R_SH, :], axis=1)
    sh_width_med = float(np.nanmedian(sh_width)) or 1e-6

    spine_vec = hip_mid - sh_mid
    spine_angle = np.degrees(np.arctan2(np.abs(spine_vec[:, 0]), np.abs(spine_vec[:, 1])))
    spine_tilt = float(np.nanmedian(spine_angle))

    nose = coords[:, NOSE, :]
    head_stability = float(np.nansum(np.linalg.norm(np.diff(nose, axis=0), axis=1)) / sh_width_med)
    head_stability_norm = round(head_stability / n * 100, 3)

    wr_y = np.nanmean(np.stack([coords[:, L_WR, 1], coords[:, R_WR, 1]], axis=1), axis=1)
    k = max(1, int(fps * 0.1)); kernel = np.ones(2 * k + 1) / (2 * k + 1)
    wr_y_s = np.convolve(np.nan_to_num(wr_y, nan=np.nanmean(wr_y)), kernel, mode='same')
    search_end = max(int(n * 0.75), 3)
    top_idx = int(np.argmin(wr_y_s[:search_end]))
    impact_window_end = min(top_idx + max(int(n * 0.5), 3), n)
    impact_idx = top_idx + int(np.argmax(wr_y_s[top_idx:impact_window_end]))
    backswing_t = top_idx / fps
    downswing_t = max(impact_idx - top_idx, 1) / fps
    top_at_start = top_idx <= 2

    if top_at_start or backswing_t <= 0.05:
        tempo = {"measurable": False,
                 "reason": "영상이 백스윙 도중부터 시작해 백스윙 구간이 없습니다.",
                 "tip": "어드레스(공 앞에 선 자세)부터 찍으면 템포 분석을 받을 수 있어요."}
    else:
        ratio = round(backswing_t / downswing_t, 2)
        desc = "느긋한" if ratio >= 3.2 else "표준적인" if ratio >= 2.5 else "빠른"
        tempo = {"measurable": True, "ratio": ratio, "desc": desc,
                 "backswing_sec": round(backswing_t, 2), "downswing_sec": round(downswing_t, 2)}

    features = {
        "tempo": tempo,
        "total_swing_sec": {"measurable": True, "value": round(n / fps, 2)},
        "spine_tilt_deg": {"measurable": True, "value": round(spine_tilt, 1),
                           "angle_dependent": True,
                           "note": "촬영 각도에 따라 값이 달라질 수 있어 비교 시 참고용."},
        "head_stability": {"measurable": True, "value": head_stability_norm,
                           "desc": ("매우 안정적" if head_stability_norm < 5 else
                                    "안정적" if head_stability_norm < 10 else "움직임 있음")},
    }

    frames_json = []
    for i in range(n):
        joints = {}
        for name, j in SAVE_JOINTS.items():
            x, y = coords[i, j]
            joints[name] = {"x": round(float(x), 4), "y": round(float(y), 4),
                            "v": round(float(vis_raw[i, j]), 3)}
        frames_json.append(joints)

    pose_data = {
        "version": "pose_v3",
        "video": {"width": w, "height": h, "fps": round(fps, 2), "frame_count": n},
        "detection_rate": round(float(np.mean(vis_raw[:, L_SH] > 0)), 3),
        "skeleton": SKELETON,
        "measure_lines": MEASURE_LINES,
        "vis_threshold": VIS_SHOW,
        "features": features,
        "swing_phases": {"top_frame": int(top_idx), "impact_frame": int(impact_idx),
                         "top_at_start": bool(top_at_start)},
        "frames": frames_json,
        "_note": "좌표는 0~1 정규화. 프론트는 x*video.width, y*video.height 로 canvas에 그림. "
                 "v<vis_threshold 관절은 미표시(clean). measure_lines로 비교모달 자동 선긋기.",
    }
    return pose_data


# ─────────────────────────────────────────────────────────────────────
# Supabase 연동 (REST + Storage)
# ─────────────────────────────────────────────────────────────────────
def claim_pending(limit):
    """pose_status='pending' 인 행을 가져온다. (오래된 것부터)
    동시에 여러 워커가 떠도 안전하도록 가져오는 즉시 'processing' 으로 마킹."""
    # 1) pending 조회 (id, video_path 만)
    url = (REST + "/analysis_logs"
           "?pose_status=eq.pending"
           "&select=id,video_path"
           "&order=created_at.asc"
           f"&limit={limit}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = r.json()
    claimed = []
    for row in rows:
        log_id = row["id"]
        # 2) pending → processing 으로 조건부 UPDATE (이미 누가 채갔으면 0행 → 스킵)
        upd = requests.patch(
            REST + f"/analysis_logs?id=eq.{log_id}&pose_status=eq.pending",
            headers={**HEADERS, "Prefer": "return=representation"},
            data=json.dumps({"pose_status": "processing"}),
            timeout=30,
        )
        if upd.ok and upd.json():
            claimed.append(row)
    return claimed


def requeue_stale():
    """processing 인 채로 STALE_PROCESSING_MIN 분 넘게 멈춘 행을 pending 으로 되돌림.
    (워커가 처리 중 죽은 경우 유실 방지)"""
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S",
                           time.gmtime(time.time() - STALE_PROCESSING_MIN * 60))
    # created_at 이 아니라 처리 시각 기준이 이상적이나, 별도 컬럼 없이 보수적으로
    # processing 전체 중 cutoff 이전 생성건을 되돌린다(드문 케이스라 충분).
    url = (REST + "/analysis_logs"
           "?pose_status=eq.processing"
           f"&created_at=lt.{cutoff}")
    try:
        requests.patch(url, headers=HEADERS,
                       data=json.dumps({"pose_status": "pending"}), timeout=30)
    except Exception as e:
        print(f"[stale] 되돌리기 실패(무시): {e}", flush=True)


def download_video(video_path, dst_path):
    """Storage 의 video_path 파일을 dst_path 로 다운로드. service_role 권한으로 직접 접근."""
    url = f"{STORAGE}/object/{VIDEO_BUCKET}/{video_path}"
    r = requests.get(url, headers={"apikey": SERVICE_KEY,
                                   "Authorization": "Bearer " + SERVICE_KEY},
                     timeout=120)
    r.raise_for_status()
    with open(dst_path, "wb") as f:
        f.write(r.content)


def upload_pose_json(log_id, pose_data):
    """pose.json 을 Storage pose/{log_id}.json 으로 업로드. 반환: pose_path."""
    pose_path = f"pose/{log_id}.json"
    body = json.dumps(pose_data, ensure_ascii=False).encode("utf-8")
    url = f"{STORAGE}/object/{POSE_BUCKET}/{pose_path}"
    # x-upsert: true → 재처리 시 덮어쓰기 허용
    r = requests.post(url, headers={"apikey": SERVICE_KEY,
                                    "Authorization": "Bearer " + SERVICE_KEY,
                                    "Content-Type": "application/json",
                                    "x-upsert": "true"},
                      data=body, timeout=60)
    r.raise_for_status()
    return pose_path


def mark(log_id, status, pose_path=None):
    payload = {"pose_status": status}
    if pose_path is not None:
        payload["pose_path"] = pose_path
    requests.patch(REST + f"/analysis_logs?id=eq.{log_id}",
                   headers=HEADERS, data=json.dumps(payload), timeout=30)


# ─────────────────────────────────────────────────────────────────────
# 처리 1건
# ─────────────────────────────────────────────────────────────────────
def process_one(row, lm, mp):
    log_id = row["id"]
    video_path = row.get("video_path")
    if not video_path:
        # 저장 영상이 아닌데 pending 으로 들어온 경우 → 대상 아님 처리
        print(f"[{log_id}] video_path 없음 → failed 처리", flush=True)
        mark(log_id, "failed")
        return
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "input.mp4")
        t0 = time.time()
        download_video(video_path, local)
        pose_data = analyze(local, lm, mp)
        if pose_data is None:
            print(f"[{log_id}] 분석 결과 없음(프레임 0) → failed", flush=True)
            mark(log_id, "failed")
            return
        pose_path = upload_pose_json(log_id, pose_data)
        mark(log_id, "done", pose_path)
        dr = pose_data.get("detection_rate", 0)
        fc = pose_data.get("video", {}).get("frame_count", 0)
        print(f"[{log_id}] ✅ done — {fc}프레임 탐지율 {dr*100:.0f}% "
              f"({time.time()-t0:.1f}s) → {pose_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────
# 메인 폴링 루프
# ─────────────────────────────────────────────────────────────────────
def main():
    print("[워커] 시작. 모델 준비 중...", flush=True)
    ensure_model()
    import mediapipe as mp
    lm = make_landmarker()  # 1회만 생성 → 상주 재사용
    print(f"[워커] 준비 완료. {POLL_INTERVAL}초 간격 폴링 시작 "
          f"(batch={BATCH_SIZE}, bucket={VIDEO_BUCKET})", flush=True)

    last_stale_check = 0
    while True:
        try:
            # 5분마다 stale processing 되돌리기
            if time.time() - last_stale_check > 300:
                requeue_stale()
                last_stale_check = time.time()

            rows = claim_pending(BATCH_SIZE)
            if not rows:
                time.sleep(POLL_INTERVAL)
                continue
            print(f"[워커] pending {len(rows)}건 처리 시작", flush=True)
            for row in rows:
                try:
                    process_one(row, lm, mp)
                except Exception as e:
                    print(f"[{row.get('id')}] ❌ 처리 실패: {e}", flush=True)
                    traceback.print_exc()
                    try:
                        mark(row["id"], "failed")
                    except Exception:
                        pass
            # 처리 직후엔 곧바로 다음 배치 확인(밀린 큐 빠르게 소진)
        except Exception as e:
            print(f"[워커] 루프 오류(계속): {e}", flush=True)
            traceback.print_exc()
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
