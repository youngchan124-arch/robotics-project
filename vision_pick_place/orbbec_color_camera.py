"""
Orbbec Astra S 같은 OpenNI2 전용(UVC 미지원) 깊이카메라의 컬러 스트림을 읽기 위한 헬퍼.

Astra S는 lsusb에는 잡히지만 Vendor Specific 프로토콜이라 /dev/video*에 노드가 생기지
않고, cv2.VideoCapture()로는 열 수 없다. OpenNI2를 통해서만 컬러 스트림을 받을 수 있는데,
시스템 패키지(libopenni2-0, /usr/lib/x86_64-linux-gnu)에는 PrimeSense/Kinect용 드라이버만
들어있고 Orbbec 전용 드라이버(liborbbec.so, 벤더 2bc5)가 빠져 있어서 Astra S를 인식하지
못한다 (openni2.Device.enumerate_uris()가 항상 빈 리스트를 반환하는 것으로 확인됨).

그래서 Orbbec 공식 "OpenNI-Linux-x64-2.3" SDK의 Redist 폴더(정상 동작하는 libOpenNI2.so +
liborbbec.so 드라이버 세트)를 이 저장소의 openni2_redist/ 에 통째로 넣어두고, 시스템
OpenNI2 대신 이걸 가리키도록 했다. pip install primesense 로 설치한 ctypes 기반 OpenNI2
파이썬 바인딩을 사용한다.

사용 전 준비:
  1) pip install primesense  (lerobot_312 conda env에 설치됨)
  2) udev 규칙 등록 (일반 사용자 권한으로 USB 장치를 열 수 있도록):
     sudo cp 61-orbbec-astra.rules /etc/udev/rules.d/
     sudo udevadm control --reload-rules && sudo udevadm trigger
     (이미 적용 완료 — mode 666으로 확인됨. USB 재연결 시 다시 확인 필요할 수 있음)

ThreadedCamera(cv2 기반, debug_camera_preview.py / 101_data_collect_using_teleop.py에서
쓰던 클래스)와 동일한 인터페이스(isOpened / read / release)를 제공하므로, 탑뷰 카메라
자리에 그대로 바꿔 끼울 수 있다.
"""

import os
import threading

import cv2
import numpy as np
from primesense import openni2

# OpenNI2 라이브러리(.so)와 드라이버(Drivers/liborbbec.so 등)가 있는 경로.
# 시스템 openni2에는 Orbbec 드라이버가 없어서, 이 저장소에 함께 담아둔 번들을 기본값으로 사용.
DEFAULT_OPENNI2_REDIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openni2_redist")

_oni_lock = threading.Lock()
_oni_initialized = False


def _ensure_openni_initialized(redist_dir=DEFAULT_OPENNI2_REDIST_DIR):
    # openni2.initialize()는 프로세스당 한 번만 호출하면 되고, 여러 카메라 인스턴스가
    # 동시에 부르면 내부 상태가 꼬일 수 있어 락으로 보호한다.
    global _oni_initialized
    with _oni_lock:
        if not _oni_initialized:
            openni2.initialize(redist_dir)
            _oni_initialized = True


class ThreadedOrbbecColorCamera:
    """OpenNI2 컬러 스트림을 백그라운드 스레드에서 계속 읽어와 최신 프레임을 즉시
    넘겨준다. 카메라 open/read가 잠깐 멈춰도 그 지연이 모터 제어 루프까지 전파되지
    않도록 하기 위함 (기존 ThreadedCamera와 동일한 설계 이유)."""

    def __init__(self, width=640, height=480, fps=30, redist_dir=DEFAULT_OPENNI2_REDIST_DIR):
        self.width = width
        self.height = height
        self.fps = fps
        self.redist_dir = redist_dir

        self.device = None
        self.stream = None

        self._lock = threading.Lock()
        self._ret = False
        self._frame = None
        self._running = True
        self._opened_event = threading.Event()
        self._open_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            _ensure_openni_initialized(self.redist_dir)
            self.device = openni2.Device.open_any()
            self.stream = self.device.create_color_stream()
            if self.stream is None:
                raise RuntimeError("이 장치는 컬러 스트림을 지원하지 않습니다 (create_color_stream() -> None).")
            try:
                self.stream.configure_mode(self.width, self.height, self.fps, openni2.PIXEL_FORMAT_RGB888)
            except Exception:
                pass  # 요청한 해상도/fps 조합을 장치가 지원하지 않으면 기본 모드로 계속 진행
            try:
                self.stream.set_mirroring_enabled(False)  # 기본 True라 좌우가 뒤집혀 나옴
            except Exception:
                pass
            self.stream.start()
        except Exception as e:
            self._open_error = e
            self._opened_event.set()
            return

        self._opened_event.set()

        while self._running:
            try:
                oni_frame = self.stream.read_frame()
            except Exception:
                continue

            h, w = oni_frame.height, oni_frame.width
            # bytes()로 즉시 복사해둬야 안전함: oni_frame이 다음 루프에서 해제(release)되면
            # 그 내부 버퍼를 가리키던 numpy view도 같이 무효화될 수 있음.
            raw = bytes(oni_frame.get_buffer_as_uint8())
            img_rgb = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            with self._lock:
                self._ret = True
                self._frame = img_bgr

    def isOpened(self):
        self._opened_event.wait(timeout=5.0)
        if self._open_error is not None:
            print(f"[Orbbec] 오픈 실패: {self._open_error}")
            return False
        return self.stream is not None

    def read(self):
        with self._lock:
            if self._frame is None:
                return self._ret, None
            return self._ret, self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass


class ThreadedOrbbecRGBDCamera:
    """Astra S의 컬러 스트림 + 깊이 스트림을 함께 읽는다.

    깊이는 mm 단위 raw uint16 값을 [depth_min_mm, depth_max_mm] 범위로 클리핑/정규화한 뒤
    컬러맵(JET)을 입혀 3채널 uint8 이미지로 만든다. LeRobot 데이터셋의 기존
    "video"(3ch uint8) 피처 스키마, 그리고 ACT가 카메라 키마다 자동으로 시각 인코더를
    붙이는 구조를 그대로 재사용하려는 목적 — raw depth(1채널, mm 단위)를 그대로 넣으려면
    데이터셋 스키마와 정책 쪽 옵저베이션 인코더를 새로 설계해야 해서 훨씬 손이 많이 간다.

    depth_min_mm/depth_max_mm은 실제 작업대까지의 거리에 맞게 조정해야 한다.
    debug_depth_top_camera_preview.py를 실행하면 현재 프레임의 유효 깊이 min/max(mm)를
    콘솔에 출력해주니, 그 값을 보고 조정할 것.

    깊이 프레임을 컬러 프레임의 화각/좌표계에 맞춰 정렬(IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    하고 시간 동기화(depth_color_sync)도 켠다 — 안 그러면 깊이 이미지가 컬러 이미지보다
    화각이 좁아 상자 위치가 서로 어긋나 보인다.
    """

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        depth_width=320,
        depth_height=240,
        depth_min_mm=350,
        depth_max_mm=800,
        redist_dir=DEFAULT_OPENNI2_REDIST_DIR,
    ):
        # depth defaults to a lower resolution than color, not the same
        # width/height - measured live on the Astra S (2026-08-26): color
        # VGA + depth VGA running together stalls hard after the very first
        # frame (read_frame() on the second stream just never returns - a
        # USB2.0 bandwidth ceiling, not a code bug), while color VGA + depth
        # QVGA runs both at a smooth ~30fps indefinitely. Depth is only ever
        # used for a colorized monitor view here, so the lower resolution
        # costs nothing that matters.
        self.width = width
        self.height = height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        self.depth_min_mm = depth_min_mm
        self.depth_max_mm = depth_max_mm
        self.redist_dir = redist_dir

        self.device = None
        self.color_stream = None
        self.depth_stream = None

        self._lock = threading.Lock()
        self._ret = False
        self._color = None
        self._depth_vis = None
        self._depth_mm = None
        self._running = True
        self._opened_event = threading.Event()
        self._open_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            _ensure_openni_initialized(self.redist_dir)
            self.device = openni2.Device.open_any()
            self.color_stream = self.device.create_color_stream()
            self.depth_stream = self.device.create_depth_stream()
            if self.color_stream is None or self.depth_stream is None:
                raise RuntimeError("이 장치는 컬러/깊이 스트림을 모두 지원해야 합니다.")

            try:
                self.color_stream.configure_mode(self.width, self.height, self.fps, openni2.PIXEL_FORMAT_RGB888)
            except Exception:
                pass  # 요청한 해상도/fps를 장치가 지원하지 않으면 기본 모드로 계속 진행
            try:
                self.depth_stream.configure_mode(
                    self.depth_width, self.depth_height, self.fps, openni2.PIXEL_FORMAT_DEPTH_1_MM
                )
            except Exception:
                pass

            # Astra S는 컬러/깊이 스트림 모두 mirroring_enabled가 기본 True로 켜져 있어서
            # (셀카 앱 등 사람이 카메라를 보고 서는 용도를 가정한 기본값으로 추정) 실제
            # 물리 배치와 좌우가 뒤집혀 나온다. 탑뷰 카메라로 쓸 때는 필요 없으므로 끈다.
            try:
                self.color_stream.set_mirroring_enabled(False)
            except Exception as e:
                print(f"[Orbbec] 경고: 컬러 스트림 미러링을 끌 수 없습니다: {e}")
            try:
                self.depth_stream.set_mirroring_enabled(False)
            except Exception as e:
                print(f"[Orbbec] 경고: 깊이 스트림 미러링을 끌 수 없습니다: {e}")

            try:
                self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
            except Exception as e:
                print(f"[Orbbec] 경고: depth-to-color 정렬(image registration)을 켤 수 없습니다: {e}")
            try:
                self.device.set_depth_color_sync_enabled(True)
            except Exception:
                pass

            self.color_stream.start()
            self.depth_stream.start()
        except Exception as e:
            self._open_error = e
            self._opened_event.set()
            return

        self._opened_event.set()

        while self._running:
            try:
                color_oni = self.color_stream.read_frame()
                depth_oni = self.depth_stream.read_frame()
            except Exception:
                continue

            ch, cw = color_oni.height, color_oni.width
            craw = bytes(color_oni.get_buffer_as_uint8())
            img_rgb = np.frombuffer(craw, dtype=np.uint8).reshape((ch, cw, 3))
            color_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            dh, dw = depth_oni.height, depth_oni.width
            draw = bytes(depth_oni.get_buffer_as_uint16())
            depth_mm = np.frombuffer(draw, dtype=np.uint16).reshape((dh, dw))
            depth_vis = self._depth_to_vis(depth_mm)

            with self._lock:
                self._ret = True
                self._color = color_bgr
                self._depth_vis = depth_vis
                self._depth_mm = depth_mm

    def _depth_to_vis(self, depth_mm):
        clipped = np.clip(depth_mm, self.depth_min_mm, self.depth_max_mm).astype(np.float32)
        norm = ((clipped - self.depth_min_mm) / (self.depth_max_mm - self.depth_min_mm) * 255.0).astype(np.uint8)
        vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        vis[depth_mm == 0] = (0, 0, 0)  # 0mm = 측정 실패(무효) 픽셀. 검은색으로 표시해 실측 근접값과 구분
        return vis

    def isOpened(self):
        self._opened_event.wait(timeout=5.0)
        if self._open_error is not None:
            print(f"[Orbbec] 오픈 실패: {self._open_error}")
            return False
        return self.color_stream is not None and self.depth_stream is not None

    def read(self):
        """반환: (ret, color_bgr, depth_vis_bgr)"""
        with self._lock:
            if self._color is None:
                return self._ret, None, None
            return self._ret, self._color.copy(), self._depth_vis.copy()

    def read_raw_depth_mm(self):
        """가장 최근 깊이 프레임의 원본 mm 값(uint16, HxW). 캘리브레이션/디버그용."""
        with self._lock:
            return None if self._depth_mm is None else self._depth_mm.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        for stream in (self.color_stream, self.depth_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
