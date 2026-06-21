"""LIVE mp4 streamer (diagnostic) -- click play and THIS PC pushes the video to the
remote browser frame-by-frame in real time, with NO client pre-buffering.

Why: a buffered file download plays smoothly off cache and hides link problems. This
instead streams an already-rendered avatar clip as live MJPEG (multipart/x-mixed-replace),
paced at the clip's native framerate via ffmpeg `-re`. The client fetches the stream,
parses each JPEG and draws it to a <canvas> (works in every modern browser, unlike
<img src=multipart> which Safari/some browsers ignore). Frames appear as they arrive, so
jitter/throughput shortfalls show up immediately -- the same real-time condition the live
avatar runs under, but with ZERO GPU/render involved. So:
  * smooth here, avatar laggy  -> the render/system is fine; the lag is the live transport.
  * laggy here too             -> the link can't carry live video at this rate (connection).

We size the MJPEG to ~1.5-2 Mbps (q:v ~12 at 512px), matching the avatar's real video
bitrate, so this is a fair test and not "MJPEG is just fatter".

Run:
    python -m scripts.stream_live --dir output/_serve --host 127.0.0.1 --port 8090
Exposed to the remote viewer over the existing Tailscale host:
    tailscale serve --bg --set-path /watch http://127.0.0.1:8090
    -> https://<machine>.<tailnet>.ts.net/watch/
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SRC_CLIP = "full_quality.mp4"  # the full-quality avatar render (512px, 25fps)
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
SERVE_DIR = "output/_serve"
BUILD = "build-5 relative-url-fix"

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>VisualLLm - live stream test</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0c0d10;color:#e7e9ee;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;
 display:flex;flex-direction:column;align-items:center;padding:20px 16px 56px}
h1{font-size:18px;margin:0 0 2px}
.tag{color:#5b6personally;font-size:11px;margin-bottom:14px}
.bar{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-bottom:14px}
button{background:#1d2128;border:1px solid #2c2f3a;color:#e7e9ee;font:inherit;
 padding:10px 16px;border-radius:9px;cursor:pointer}
button.play{background:#1f6feb;border-color:#1f6feb}
button.stop{background:#2a1416;border-color:#5a2630}
#stage{width:min(92vw,520px);aspect-ratio:1/1;background:#000;border:1px solid #23262e;
 border-radius:12px;overflow:hidden}
#c{width:100%;height:100%;object-fit:contain;background:#000;display:block}
#status{margin-top:12px;font-size:15px;font-weight:600;color:#7ee787;min-height:1.4em;text-align:center}
#err{margin-top:8px;color:#ff7b72;font-size:13px;max-width:90vw;white-space:pre-wrap;text-align:center}
.verdict{max-width:720px;margin:18px auto 0;padding:12px 16px;background:#12161d;
 border:1px solid #243044;border-radius:10px;color:#c4ccd9;font-size:13px}
</style></head><body>
<h1>VisualLLm - live video stream test</h1>
<div class=tag>BUILD_TAG</div>
<div class=bar>
  <button class=play onclick="go(512,12)">&#9654; 512px (current avatar)</button>
  <button class=play onclick="go(320,8)">&#9654; 320px (the fix)</button>
  <button class=stop onclick="stp()">&#9632; Stop</button>
</div>
<div id=stage><canvas id=c width=512 height=512></canvas></div>
<div id=status>starting automatically...</div>
<div id=err></div>
<div class=verdict><b>Read it:</b> the status line shows live FPS. Smooth ~24fps here +
laggy avatar = fix the transport. Stutter/freeze/low-fps here = the connection.</div>
<script>
var c=document.getElementById('c'),ctx=c.getContext('2d'),
    st=document.getElementById('status'),er=document.getElementById('err');
var ctrl=null,timer=null,t0=0,fcount=0;
window.onerror=function(m,s,l){er.textContent='JS error: '+m+' @'+l;return false;};
function S(t){st.textContent=t;}
function stp(){
  if(ctrl){try{ctrl.abort();}catch(e){}ctrl=null;}
  if(timer){clearInterval(timer);timer=null;}
  S('stopped');
}
function cat(a,b){var o=new Uint8Array(a.length+b.length);o.set(a);o.set(b,a.length);return o;}
async function go(w,q){
  stp();er.textContent='';
  c.width=w;c.height=w;ctx.fillStyle='#111';ctx.fillRect(0,0,w,w);
  t0=Date.now();fcount=0;S('connecting to /stream ...');
  if(!window.fetch||!window.ReadableStream){er.textContent='this browser has no fetch-stream support';return;}
  ctrl=new AbortController();
  timer=setInterval(function(){var s=(Date.now()-t0)/1000;
    S('LIVE '+w+'px  '+s.toFixed(0)+'s  '+fcount+' frames  '+(fcount/Math.max(1,s)).toFixed(1)+' fps');},500);
  // IMPORTANT: build the URL relative to THIS page's directory (e.g. /watch/), not
  // root-relative '/stream' -- under tailscale '/' is proxied to the pipeline, so a
  // leading slash would hit the wrong backend and 502.
  var base=location.pathname.replace(/[^/]*$/,'');
  var url=base+'stream?w='+w+'&q='+q+'&t='+Date.now();
  S('connecting to '+url+' ...');
  var resp;
  try{resp=await fetch(url,{signal:ctrl.signal});}
  catch(e){clearInterval(timer);timer=null;er.textContent='fetch failed: '+e;return;}
  if(!resp.ok){clearInterval(timer);timer=null;er.textContent='HTTP '+resp.status+' from '+url;return;}
  if(!resp.body){clearInterval(timer);timer=null;er.textContent='no resp.body (streaming unsupported)';return;}
  S('connected, waiting for frames...');
  var reader=resp.body.getReader(),dec=new TextDecoder('latin1'),buf=new Uint8Array(0);
  while(true){
    var r;try{r=await reader.read();}catch(e){break;}
    if(r.done)break;
    buf=cat(buf,r.value);
    while(true){
      var head=dec.decode(buf.subarray(0,Math.min(buf.length,512)));
      var hend=head.indexOf('\\r\\n\\r\\n');
      if(hend<0)break;
      var m=/content-length:\\s*(\\d+)/i.exec(head.slice(0,hend));
      if(!m)break;
      var ln=parseInt(m[1],10),bs=hend+4,need=bs+ln;
      if(buf.length<need)break;
      var jpg=buf.slice(bs,bs+ln);buf=buf.slice(need);
      try{var bmp=await createImageBitmap(new Blob([jpg],{type:'image/jpeg'}));
        ctx.drawImage(bmp,0,0,c.width,c.height);if(bmp.close)bmp.close();fcount++;}
      catch(e){er.textContent='decode err: '+e;}
    }
  }
}
// auto-start so we don't depend on a click working
go(512,12);
</script></body></html>"""
PAGE = PAGE.replace("BUILD_TAG", BUILD).replace("#5b6personally", "#5b6270")


class Handler(BaseHTTPRequestHandler):
    serve_dir = SERVE_DIR

    def _page(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, qs: dict) -> None:
        src = os.path.join(self.serve_dir, SRC_CLIP)
        if not os.path.isfile(src):
            self.send_error(404, "source clip missing")
            return
        try:
            w = max(64, min(1024, int(qs.get("w", ["512"])[0])))
        except ValueError:
            w = 512
        try:
            q = max(2, min(31, int(qs.get("q", ["12"])[0])))
        except ValueError:
            q = 12
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-re", "-stream_loop", "-1", "-i", src,
            "-vf", f"scale={w}:{w}:flags=bicubic",
            "-f", "mpjpeg", "-q:v", str(q), "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=ffmpeg")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # browser stopped/navigated away -- expected
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._page()
            return
        if parsed.path == "/stream":
            self._stream(urllib.parse.parse_qs(parsed.query))
            return
        self.send_error(404)

    def log_message(self, fmt, *args):  # show requests (helps debug the remote)
        try:
            print("REQ", self.address_string(), self.command, self.path)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Live MJPEG mp4 streamer (diagnostic).")
    ap.add_argument("--dir", default=SERVE_DIR)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    Handler.serve_dir = os.path.abspath(args.dir)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LIVE streamer ({BUILD}) on http://{args.host}:{args.port}/  src={os.path.join(Handler.serve_dir, SRC_CLIP)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
