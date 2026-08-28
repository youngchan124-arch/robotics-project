"""Browser-based fallback for zeroshot_viewer.py's live window.

2026-08-28: the user reported the cv2 window still looked frozen even after
two rounds of real fixes (decoupling video from the slow label-render cycle,
then adding an unmissable ticking clock+frame-counter+moving-dot overlay).
Independently verified via md5sum that the underlying published frame
(FULL_VIEW_PATH, written by zeroshot_viewer.py's main loop every tick) WAS
genuinely changing every second - so the pipeline itself is live. The most
likely remaining explanation is the cv2/Qt window itself not repainting on
screen: every launch log for this project's cv2 GUI scripts (this one,
astra_s_live.py, camera_hub.py) prints "Ignoring XDG_SESSION_TYPE=wayland on
Gnome" - running a Qt5 highgui window through XWayland compatibility under a
Wayland GNOME session is a known source of stale/non-repainting windows
(some window managers only recomposite on focus/expose events, which a
background-launched `cv2.imshow` loop may never properly trigger).

Rather than keep chasing that (unverifiable from here - this session has no
way to see the user's actual screen), this sidesteps the whole X11/Wayland/
Qt window pipeline: a tiny local HTTP server serves zeroshot_viewer.py's
already-published FULL_VIEW_PATH image, wrapped in an HTML page whose <img>
tag re-fetches with a cache-busting query param on a JS interval. A browser
reliably repaints on every image load - no window-manager/compositor
dependency at all. Does NOT read the camera or run any model itself - purely
serves whatever zeroshot_viewer.py is already writing to disk, so it must be
run ALONGSIDE that script (or astra_s_live.py alone, as a plain RGB fallback
- see NO_VIEWER_FALLBACK below), not instead of it.

Run: `uv run python3 zeroshot_http_view.py` (no GPU/model deps at all, any
venv with a stdlib http.server works), then open http://localhost:8899/ in
any browser on this machine.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import time

import config

PORT = 8899
FULL_VIEW_PATH = "/tmp/vsp_zeroshot_viewer_full.png"  # must match zeroshot_viewer.py's own constant
# If zeroshot_viewer.py isn't running (models not loaded, or it crashed),
# fall back to the plain Astra RGB publish so this page still shows SOME
# live image instead of a hard error - astra_s_live.py alone is enough for
# that, no GPU/models needed for this fallback path.
NO_VIEWER_FALLBACK_PATH = config.ASTRA_RGB_FRAME_PATH

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Zeroshot live view</title>
<style>
  body {{ background:#111; color:#0f0; font-family: monospace; text-align:center; margin:0; padding:10px; }}
  img {{ max-width:100%; border:2px solid #0f0; }}
  #status {{ margin-top:8px; font-size:14px; }}
</style></head>
<body>
  <img id="frame" src="/frame.png?t=0">
  <div id="status">loading...</div>
  <script>
    const img = document.getElementById('frame');
    const status = document.getElementById('status');
    let n = 0;
    setInterval(() => {{
      n++;
      const t = Date.now();
      const probe = new Image();
      probe.onload = () => {{ img.src = probe.src; status.textContent = 'refresh #' + n + ' @ ' + new Date(t).toLocaleTimeString(); }};
      probe.onerror = () => {{ status.textContent = 'refresh #' + n + ' FAILED - is zeroshot_http_view.py still running?'; }};
      probe.src = '/frame.png?t=' + t;
    }}, {refresh_ms});
  </script>
</body></html>
"""

REFRESH_MS = 400  # snappier than the underlying ~1s file-publish rate on purpose - a browser fetch is cheap either way


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet - one log line per refresh tick would be pure noise
        pass

    def do_GET(self):
        if self.path.startswith("/frame.png"):
            path = FULL_VIEW_PATH if os.path.exists(FULL_VIEW_PATH) else NO_VIEWER_FALLBACK_PATH
            if not os.path.exists(path):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"no frame published yet - is astra_s_live.py running?")
                return
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
        else:
            body = PAGE.format(refresh_ms=REFRESH_MS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"[zeroshot_http_view] serving http://localhost:{PORT}/  (Ctrl-C to stop)")
        print(f"[zeroshot_http_view] source: {FULL_VIEW_PATH} (falls back to {NO_VIEWER_FALLBACK_PATH} if missing)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
