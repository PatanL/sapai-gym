"""Bounded official-client replay capture, with evidence-first completion.

Requires the existing 1920x1080 local browser. Never starts a live match. Every
navigation click is separate from recorded human actions. Completion means all
expected shop turns were observed plus return to menu, NOT full state accuracy.
"""
from __future__ import annotations

import base64
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import cv2
import numpy as np

from .archive import atomic_json, immutable_json, lock, sha_file, utc


def response_shape(value, depth=0):
    """Replay-response diagnostics without values, credentials or headers."""
    if depth >= 5:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(k):response_shape(v,depth+1) for k,v in list(value.items())[:40]}
    if isinstance(value, list):
        return {"type":"list","count":len(value),"firstShape":response_shape(value[0],depth+1) if value else None}
    if isinstance(value,str) and value.lstrip().startswith(("{","[")):
        try:return {"encodedJSON":response_shape(json.loads(value),depth+1)}
        except ValueError:pass
    return type(value).__name__


class ReplayScreen:
    def __init__(self, root, profile):
        from sapai_gym1.deployment import stat_ocr
        self.ocr = stat_ocr
        self.digits = stat_ocr.DigitTemplateRecognizer(Path(root) / "runs/live-vision/stat-digits-v1.npz")
        self.profile = json.loads(Path(profile).read_text())
        self.references = {}
        for name, ref in self.profile["references"].items():
            path = Path(root) / ref["file"]
            if sha_file(path) != ref["sha256"]:
                raise ValueError(f"UI reference changed: {name}")
            image = cv2.imread(str(path))
            x1, y1, x2, y2 = ref["box"]
            self.references[name] = image[y1:y2, x1:x2]
        cv2.setNumThreads(1)

    def score(self, im, name):
        if im is None or im.shape[:2] != (1080, 1920):
            return -1.0
        x1, y1, x2, y2 = self.profile["references"][name]["box"]
        if self.profile["references"][name].get("mask") == "white":
            a = np.all(im[y1:y2,x1:x2] > 220,axis=2)
            b = np.all(self.references[name] > 220,axis=2)
            return float((a & b).sum() / max(1,(a | b).sum()))
        return float(cv2.matchTemplate(im[y1:y2, x1:x2], self.references[name], cv2.TM_CCOEFF_NORMED)[0, 0])

    def visible(self, im, name):
        return self.score(im, name) >= self.profile["references"][name].get("threshold", .92)

    def hud(self, im, box):
        x1, y1, x2, y2 = box
        crop = im[y1:y2, x1:x2]
        if float(np.all(crop > 200, axis=2).mean()) < .20:
            return None
        mask = np.all(crop < 100, axis=2).astype("uint8")
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        glyphs = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if h >= 12 and area >= 30 and x > 0 and y > 0 and x+w < mask.shape[1] and y+h < mask.shape[0]:
                try:
                    digit, confidence, _ = self.digits._digit(self.ocr.normalize_glyph(labels == k))
                except ValueError:
                    return None
                glyphs.append((int(x), digit, confidence))
        glyphs.sort()
        if not 1 <= len(glyphs) <= 2:
            return None
        return {"value": int("".join(str(x[1]) for x in glyphs)), "confidence": min(x[2] for x in glyphs)}

    def inspect(self, im):
        shop = self.visible(im, "shop_controls")
        return {"shop": shop, "turn": self.hud(im, (283,17,329,50)) if shop else None,
                "gold": self.hud(im, (60,17,106,50)) if shop else None,
                "lives": self.hud(im, (178,17,220,50)) if shop else None,
                "menu": self.visible(im, "main_history"), "skip": self.visible(im, "skip_battle"),
                "play": self.visible(im, "play") if shop else False}


def coverage(expected, frames, menu_seen):
    counts = Counter(f["observation"]["turn"]["value"] for f in frames
                     if f["observation"].get("shop") and f["observation"].get("turn"))
    missing = [turn for turn in expected if counts[turn] < 2]
    return {"expectedTurns": expected, "seenTurns": sorted(counts), "framesByTurn": dict(counts),
            "missingOrSingleFrameTurns": missing, "returnToMenuObserved": menu_seen,
            "turnCoverageComplete": bool(expected) and not missing and menu_seen,
            "actionCoverageValidated": False, "trainingApproved": False}


class ReplayWorker:
    def __init__(self, page, screen, output, reload_client=False):
        self.page, self.screen, self.output = page, screen, Path(output)
        self.navigation = []
        self.started = time.monotonic()
        self.reload_client = reload_client

    def image(self):
        raw = self.page.screenshot(timeout=20000)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image.shape[:2] != (1080,1920):
            raise ValueError("Unsupported browser/canvas resolution; no blind scaling")
        return image

    def click(self, x, y, reason, wait=600):
        # Unity polls input during rendered frames. An instantaneous click plus
        # immediate move can land at the wrong coordinates on a throttled CPU.
        self.page.mouse.click(x, y, delay=110)
        self.navigation.append({"seconds": time.monotonic()-self.started, "x": x, "y": y, "reason": reason})
        self.page.wait_for_timeout(wait)
        if reason != "focus_replay_code":
            self.page.mouse.move(1850,920)

    def require(self, name, timeout=10):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            im = self.image()
            if self.screen.visible(im, name):
                return im
            self.page.wait_for_timeout(350)
        self.page.screenshot(path=str(self.output / f"failed-{name}.png"))
        raise RuntimeError(f"Expected UI control not found: {name}")

    def load(self, code):
        im = self.image()
        if self.reload_client:
            if not (self.screen.visible(im,'main_history') or self.screen.visible(im,'shop_controls')):
                raise RuntimeError('Refusing reload outside verified menu/replay')
            self.navigation.append({'seconds':time.monotonic()-self.started,'reason':'reload_verified_replay_client'})
            # The dashboard and this worker have distinct Playwright CDP
            # connections. A Unity beforeunload prompt makes both drivers race
            # to dismiss the same dialog. In a verified menu/replay only, remove
            # that leave-page prompt before the explicitly requested reload.
            self.page.evaluate("window.onbeforeunload=null; window.addEventListener('beforeunload',e=>{e.stopImmediatePropagation(); e.returnValue='';}, {capture:true,once:true})")
            self.page.reload(wait_until='domcontentloaded',timeout=45000)
            im=self.require('main_history',timeout=90)
        if self.screen.visible(im, "shop_controls"):
            self.click(846,1036,"stop_previous_replay")
            im = self.require("main_history")
        if self.screen.visible(im, "main_history"):
            self.click(960,767,"open_history")
            im = self.require("history_replays")
        if self.screen.visible(im, "history_replays"):
            self.click(960,482,"open_replays",1000)
            im = self.image()
        # A replay-list scrollbar and the verified Watch Replay button are
        # required. No fallback click into an unrecognized live-game screen.
        if not self.screen.visible(im, "watch_replay"):
            mask = cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV), np.array([5,150,140]),np.array([25,255,255]))
            _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
            thumbs = [s for s in stats[1:] if s[0] > 1850 and s[2] > 25 and s[3] > 80]
            if len(thumbs) != 1:
                raise RuntimeError("Replay-list scrollbar not uniquely identified")
            x,y,w,h,_ = thumbs[0]
            self.page.mouse.move(int(x+w/2),int(y+h/2)); self.page.mouse.down()
            self.page.wait_for_timeout(200)
            self.page.mouse.move(int(x+w/2),1060,steps=10)
            self.page.wait_for_timeout(700); self.page.mouse.up()
            self.page.wait_for_timeout(600)
        self.require("watch_replay")
        self.click(970,910,"focus_replay_code",400)
        self.page.keyboard.press("Control+A",delay=90)
        self.page.keyboard.type(json.dumps(code,separators=(",",":")),delay=35)
        self.page.wait_for_timeout(700)
        self.page.screenshot(path=str(self.output / "code-entered.png"))
        self.click(980,1012,"load_requested_replay",2000)
        self.require("shop_controls",timeout=30)
        # Fresh loads were observed stalled at turn 0. Explicit first-turn
        # navigation is necessary; verify the first-row button before clicking.
        self.click(960,1030,"open_goto",700)
        self.require("goto_first")
        self.page.screenshot(path=str(self.output / "goto-first.png"))

    def capture(self, job, stop, max_seconds=900):
        self.output.mkdir(parents=True, exist_ok=False)
        immutable_json(self.output / "JOB.json", job)
        frames, menu_seen, reason = [], False, "duration_limit"
        latest = {"image": None, "frame": None}
        cdp = self.page.context.new_cdp_session(self.page)
        last_saved = [0.]
        total_bytes = [0]
        network = []
        self.started = time.monotonic()
        def on_response(response):
            url=urlsplit(response.url)
            if url.hostname!='api.teamwood.games' or url.path!='/0.48/api/playback/participation':
                return
            record={"path":url.path,"status":response.status,"seconds":time.monotonic()-self.started}
            try:record['shape']=response_shape(response.json())
            except Exception as exc:record['shapeError']=type(exc).__name__
            network.append(record)
            atomic_json(self.output/'PLAYBACK_RESPONSE_SHAPES.json',network)
        self.page.on('response',on_response)
        def save(status):
            atomic_json(self.output / "INDEX.json", {"schema":"sap-replay-capture-v1", "job": job,
                "frames":frames,"navigation":self.navigation,"seconds":time.monotonic()-self.started,
                "endReason":status,"coverage":coverage(job["expectedShopTurns"],frames,menu_seen),
                "captureSourceSHA256":sha_file(__file__),
                "bytes":total_bytes[0],"trainingApproved":False})
        def on_frame(event):
            cdp.send("Page.screencastFrameAck", {"sessionId":event["sessionId"]})
            now = time.monotonic()
            if now-last_saved[0] < .20 or len(frames) >= 5000:
                return
            raw = base64.b64decode(event["data"])
            im = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if im is None or im.shape[:2] != (1080,1920):
                return
            name = f"frame-{len(frames):05d}.jpg"
            (self.output / name).write_bytes(raw)
            row = {"frame":name,"sha256":hashlib.sha256(raw).hexdigest(),"seconds":now-self.started,
                   "browserTimestamp":event["metadata"].get("timestamp"),"observation":self.screen.inspect(im)}
            frames.append(row)
            latest.update(image=im,frame=row)
            total_bytes[0] += len(raw)
            last_saved[0] = now
        try:
            self.load(job["code"])
            cdp.on("Page.screencastFrame", on_frame)
            cdp.send("Page.startScreencast",{"format":"jpeg","quality":94,"maxWidth":1920,"maxHeight":1080,"everyNthFrame":1})
            playback_started = time.monotonic()
            self.click(486,415,"start_verified_first_turn",120)
            last_input, report_at = time.monotonic(),time.monotonic()
            max_turn, skipped_battle, last_progress, prior_digest = 0,False,time.monotonic(),None
            opening_retries = 0
            while time.monotonic()-playback_started < max_seconds and not stop["requested"]:
                self.page.wait_for_timeout(120)
                now = time.monotonic()
                row = latest["frame"]
                if row is None:
                    continue
                obs = row["observation"]
                if row["sha256"] != prior_digest:
                    last_progress, prior_digest = now,row["sha256"]
                if obs["turn"]:
                    max_turn = max(max_turn,obs["turn"]["value"])
                if obs["menu"] and max_turn:
                    menu_seen,reason = True,"replay_returned_to_menu"
                    break
                # Some new replay loads show turn0 even after selecting GOTO1.
                # Retry only before ANY positive turn was observed, with the
                # actual HUD still at0/10gold and no team stat badges. Never
                # restart a replay that has begun making player decisions.
                if (max_turn == 0 and obs["shop"] and obs["turn"] and obs["turn"]["value"] == 0
                        and obs["gold"] and obs["gold"]["value"] == 10 and opening_retries < 2
                        and now-playback_started > 12+25*opening_retries):
                    has_team = False
                    for slot in range(5):
                        try:
                            self.screen.digits.read_pair(latest["image"],(384+144*slot)/1920,shop=False)
                            has_team = True
                        except ValueError:
                            pass
                    if not has_team:
                        if not obs["play"]:
                            self.click(1075,1033,"pause_stalled_turn0_before_retry",300)
                        self.click(960,1030,"retry_verified_turn0_goto",500)
                        self.require("goto_first")
                        self.click(486,415,"retry_verified_first_turn",500)
                        opening_retries += 1
                        last_input = time.monotonic()
                        continue
                if obs["shop"]:
                    skipped_battle = False
                    if obs["play"] and now-last_input > 3:
                        self.click(1075,1033,"resume_verified_play_button",80);last_input=now
                elif obs["skip"] and now-last_input > 2:
                    self.click(1194,55,"skip_verified_replay_combat",80)
                    last_input,skipped_battle = now,True
                elif skipped_battle and now-last_input > 6:
                    self.click(1650,650,"continue_after_replay_combat",80);last_input=now
                if max_turn == 0 and now-playback_started > 90:
                    reason="opening_stall";break
                if now-last_progress > 60:
                    reason="no_new_visual_frames";break
                if len(frames) >= 5000 or total_bytes[0] > 2_000_000_000:
                    reason="capture_storage_bound";break
                if now-report_at > 15:
                    save("running")
                    print(json.dumps({"job":job["participationId"],"frames":len(frames),"maxTurn":max_turn,"seconds":now-self.started}),flush=True)
                    report_at=now
            if stop["requested"]:
                reason="external_stop"
        except Exception as exc:
            reason=f"failed:{type(exc).__name__}:{exc}"
            raise
        finally:
            self.page.remove_listener('response',on_response)
            try:
                cdp.send("Page.stopScreencast")
                cdp.detach()
                self.page.screenshot(path=str(self.output / "final.png"),timeout=20000)
            finally:
                save(reason)
        return coverage(job["expectedShopTurns"],frames,menu_seen)


@contextmanager
def quiet_dashboard():
    """Avoid duplicate PNG encoding; restore the preexisting refresh state."""
    with urlopen("http://127.0.0.1:8770/status.json",timeout=10) as response:
        was_paused=json.load(response)["paused"]
    def command(name):
        with urlopen(Request("http://127.0.0.1:8770/"+name,data=b"{}",headers={"Content-Type":"application/json"}),timeout=10) as response:
            response.read()
    if not was_paused:
        command("pause")
    try:
        yield
    finally:
        if not was_paused:
            command("resume")


def run_queue(root, corpus, output, profile, max_jobs=4, participation_ids=None, reload_client=False):
    from playwright.sync_api import sync_playwright
    root,corpus,output = Path(root),Path(corpus),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    status = json.loads((corpus / "STATUS.json").read_text())
    jobs = []
    for match in status["matches"]:
        # Development of the parser uses TRAIN only. Holdout archive is retained
        # unopened by this visual development worker.
        if match["split"] != "train":
            continue
        path = corpus / match["directory"] / "EVIDENCE.json"
        if sha_file(path) != match["evidenceSHA256"]:
            raise ValueError("Archived evidence changed")
        evidence = json.loads(path.read_text())
        for p in evidence["perspectives"]:
            if participation_ids and p['participationId'] not in participation_ids:
                continue
            jobs.append({"matchId":match["matchId"],"replayId":match["replayId"],"participationId":p["participationId"],
                         "name":p["name"],"code":p["replayCode"],"expectedShopTurns":p["expectedShopTurns"],
                         "evidenceSHA256":match["evidenceSHA256"],"split":match["split"],"trainingApproved":False})
        if len(jobs) >= max_jobs:
            break
    jobs = jobs[:max_jobs]
    immutable_json(output / "QUEUE.json", {"schema":"sap-visual-queue-v1","jobs":jobs,
                  "profileSHA256":sha_file(profile),"trainingApproved":False,
                  **({'reloadClient':True} if reload_client else {})})
    stop={"requested":False}
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.update(requested=True))
    results=[]
    with lock(output / "worker.lock"), lock(root / "runs/replay-browser-worker.lock"), quiet_dashboard():
        screen=ReplayScreen(root,profile)
        with sync_playwright() as p:
            browser=p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            pages=[page for context in browser.contexts for page in context.pages if urlsplit(page.url).hostname=="html-classic.itch.zone"]
            if len(pages)!=1:
                raise RuntimeError("Expected exactly one official SAP page")
            page=pages[0]
            # The persistent dashboard is the dialog owner. Without a listener
            # this second Playwright connection also auto-dismisses dialogs and
            # can crash its driver with "No dialog is showing".
            page.on('dialog',lambda _dialog: None)
            page.bring_to_front()
            for job in jobs:
                prior=output / job["participationId"] / "RESULT.json"
                if prior.exists():
                    existing=json.loads(prior.read_text())
                    if existing.get("coverage",{}).get("turnCoverageComplete"):
                        results.append(existing);continue
                    attempts=list(prior.parent.glob('attempt-*/INDEX.json'))
                    if len(attempts)>=2:
                        # Keep bad replays from indefinitely blocking the next
                        # match. No fabricated replacement for their evidence.
                        results.append({**existing,'quarantined':True,'reason':'two_incomplete_capture_attempts'})
                        continue
                if stop["requested"]:
                    break
                attempt=output / job["participationId"] / ("attempt-"+uuid4().hex[:10])
                worker=ReplayWorker(page,screen,attempt,reload_client=reload_client)
                result={"job":job,"attempt":str(attempt.relative_to(output)),"startedAt":utc()}
                try:
                    result["coverage"]=worker.capture(job,stop)
                except Exception as exc:
                    result["error"]=f"{type(exc).__name__}: {exc}"
                result["finishedAt"]=utc()
                atomic_json(prior,result);results.append(result)
                atomic_json(output / "STATUS.json",{"results":results,"queued":len(jobs),"phase":"running",
                    "updatedAt":utc(),"trainingApproved":False})
                # Unknown browser state must not cascade into blind navigation.
                if "error" in result or not result.get("coverage",{}).get("returnToMenuObserved"):
                    im=worker.image()
                    if screen.visible(im,'shop_controls'):
                        worker.click(846,1036,'return_from_incomplete_replay')
                        worker.require('main_history')
                    elif not screen.visible(im,'main_history'):
                        break
            atomic_json(output / "STATUS.json",{"results":results,"queued":len(jobs),
                "phase":("finished" if all(r.get('coverage',{}).get('turnCoverageComplete') for r in results)
                          else "finished_with_quarantine") if len(results)==len(jobs) else "paused_for_review",
                "updatedAt":utc(),"trainingApproved":False})
            # Deliberately leave the user's persistent browser running.
