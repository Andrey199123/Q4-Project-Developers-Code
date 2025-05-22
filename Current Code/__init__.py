import threading
import time

import cv2
from flask import Blueprint, Response, render_template
import robot_api.camera.detection

video = Blueprint("video", __name__)


class CameraEvent(object):
    def __init__(self):
        self.events = {}

    def wait(self):
        ident = threading.get_ident()

        if not ident in self.events:
            self.events[ident] = [threading.Event(), time.time()]

        return self.events[ident][0].wait()

    def set(self):
        now = time.time()
        remove = None

        for ident, event in self.events.items():
            if not event[0].is_set():
                event[0].set()
                event[1] = now
            elif now - event[1] > 5:
                remove = ident

        if remove:
            del self.events[remove]

    def clear(self):
        self.events[threading.get_ident()][0].clear()


class Camera:
    frame = None
    raw_frame = None
    processed_frame = None
    thread = None
    image_processing_thread = None
    last_access = 0
    event = CameraEvent()

    def __init__(self):
        if Camera.thread == None:
            Camera.last_access = time.time()

            Camera.thread = threading.Thread(target=self._thread)
            Camera.image_processing_thread = threading.Thread(
                target=self._image_processing_thread
            )
            Camera.thread.start()
            Camera.image_processing_thread.start()

            Camera.event.wait()

    def get_frame(self, include_overlay=False):
        Camera.last_access = time.time()

        Camera.event.wait()
        Camera.event.clear()

        if include_overlay:
            return Camera.processed_frame
        else:
            return Camera.frame

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError("Could not start camera.")

        while True:
            _, img = camera.read()
            yield [cv2.imencode(".jpg", img)[1].tobytes(), img]

    @classmethod
    def _thread(self):
        frames_iterator = self.frames()

        for frame in frames_iterator:
            self.frame, self.raw_frame = frame
            Camera.event.set()
            time.sleep(0)

            if time.time() - Camera.last_access > 10:
                frames_iterator.close()
                break

    @classmethod
    def _image_processing_thread(self):
        i = 0

        while True:
            Camera.event.wait()

            #            if i == 0:
            self.processed_frame = cv2.imencode(
                ".jpg", detection.process_frame(self.raw_frame)
            )[1].tobytes()


#                i = (i + 1) % 4


def gen(camera, include_overlay=False):
    while True:
        frame = camera.get_frame(include_overlay)

        if frame is None:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"


@video.route("/feed")
def video_feed():
    return Response(
        gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@video.route("/feed_overlay")
def video_overlay():
    return Response(
        gen(Camera(), include_overlay=True),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


from robot_api import app

app.register_blueprint(video, url_prefix="/video")

print("Successfully registered the video feed blueprint")
