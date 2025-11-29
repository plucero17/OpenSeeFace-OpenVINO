import copy
import os
import sys
import argparse
import traceback
import gc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings (FPS still need to be set separately)", default=None)
parser.add_argument("-W", "--width", type=int, help="Set camera width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set camera height", default=360)
parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--device", type=lambda s: s.upper(), help="Select OpenVINO device", default="GPU", choices=["CPU", "GPU", "NPU"])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the OpenVINO model files", default=None)
parser.add_argument("--preproc-backend", help="Select preprocessing backend for detector/landmark/gaze models", default="opencv", choices=["opencv", "openvino"])
parser.add_argument("--pin-cores", help="Comma-separated list of CPU core indices to pin the process to", default=None)
if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

GC_INTERVAL_SECONDS = 5.0

# Set CPU Priority and CPU affinity
psutil_process = None
priority_value = getattr(args, "priority", None)
pin_core_value = args.pin_cores
if priority_value is not None or pin_core_value:
    try:
        import psutil
        psutil_process = psutil.Process(os.getpid())
    except ImportError:
        if args.silent == 0:
            print("psutil is required for priority/affinity controls but is not installed.")
        psutil_process = None

if psutil_process is not None:
    if priority_value is not None:
        PSUTIL_CLASSES = [
            psutil.IDLE_PRIORITY_CLASS, 
            psutil.BELOW_NORMAL_PRIORITY_CLASS, 
            psutil.NORMAL_PRIORITY_CLASS, 
            psutil.ABOVE_NORMAL_PRIORITY_CLASS, 
            psutil.HIGH_PRIORITY_CLASS, 
            psutil.REALTIME_PRIORITY_CLASS
        ]
        psutil_process.nice(PSUTIL_CLASSES[priority_value])

    if pin_core_value:
        try:
            core_indices = [int(core.strip()) for core in pin_core_value.split(",") if core.strip() != ""]
            if len(core_indices) > 0:
                psutil_process.cpu_affinity(core_indices)
        except Exception as e:
            if args.silent == 0:
                print(f"Failed to set CPU affinity ({e}). Proceeding without pinning.")

def list_all_cameras(cap):
    info = cap.get_info()
    print("Available cameras:")
    for cam in info:
        print(f"{cam['index']}: {cam['name']}")

def list_all_dcaps(cap, dcap_idx):
    info = cap.get_info()
    unit = 10000000.;

    VIDEO_FORMATS = {
        0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420",
        201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2",
        302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264",
    }

    for cam in info:
        if dcap_idx == -1:
            print(f"{cam['index']}: {cam['name']}")
        if dcap_idx != -1 and dcap_idx != cam['index']:
            continue
        for caps in cam['caps']:
            format = caps['format']
            if caps['format'] in VIDEO_FORMATS:
                format = VIDEO_FORMATS[caps['format']]
            if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.2f}-{unit/caps['minInterval']:.2f} Format: {format}")
            else:
                print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.2f}-{unit/caps['minInterval']:.2f} Format: {format}")

# Display Camera or DCaps Devices
if os.name == 'nt' and (args.list_cameras > 0 or args.list_dcaps is not None):
    import cam_utils.dshowcapture as dshowcapture
    cap = dshowcapture.DShowCapture()
    if args.list_dcaps is not None:
        list_all_dcaps(cap, dcap_idx=args.list_dcaps)
    elif args.list_cameras > 0:
        list_all_cameras(cap)
    cap.destroy_capture()
    sys.exit(0)

import numpy as np
import time
import cv2
cv2.setNumThreads(max(1, args.max_threads))
import socket
import struct
from cam_utils.input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from ov_backend.tracker import Tracker

target_ip = args.ip
target_port = args.port

fps = args.fps
dcap = None
use_dshowcapture_flag = False

if os.name == 'nt':
    dcap = args.dcap
    use_dshowcapture_flag = True if args.use_dshowcapture == 1 else False

    input_reader = InputReader(
        args.capture, 
        args.width, 
        args.height, 
        fps, 
        use_dshowcapture=use_dshowcapture_flag, 
        dcap=dcap,
    )

    if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
        fps = min(fps, input_reader.device.get_fps())
else:
    input_reader = InputReader(args.capture, args.width, args.height, fps)

if type(input_reader.reader) == VideoReader:
    fps = 0

first = True
height = 0
width = 0
tracker = None
sock = None
total_tracking_time = 0.0
tracking_time = 0.0
tracking_frames = 0
frame_count = 0

features = [
    "eye_l", "eye_r", 
    "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", 
    "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

is_camera = args.capture == str(try_int(args.capture))

try:
    attempt = 0
    frame_time = time.perf_counter()
    target_duration = 0
    last_gc_collect = frame_time

    if fps > 0:
        target_duration = 1. / float(fps)

    need_reinit = 0
    failures = 0
    source_name = input_reader.name

    while input_reader.is_open():
        if not input_reader.is_open() or need_reinit == 1:

            input_reader = InputReader(
                args.capture,
                args.width,
                args.height,
                fps,
                use_dshowcapture=use_dshowcapture_flag,
                dcap=dcap
            )

            if input_reader.name != source_name:
                print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                sys.exit(1)

            need_reinit = 2
            time.sleep(0.02)
            continue

        if not input_reader.is_ready():
            time.sleep(0.02)
            continue

        ret, frame = input_reader.read()
        if ret and args.mirror_input:
            frame = cv2.flip(frame, 1)

        if not ret:
            if is_camera:
                attempt += 1
                if attempt > 30:
                    break
                else:
                    time.sleep(0.02)
                    if attempt == 3:
                        need_reinit = 1
                    continue
            else:
                break;

        attempt = 0
        need_reinit = 0
        frame_count += 1
        now = time.time()

        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            tracker = Tracker(
                width,
                height,
                threshold=args.threshold,
                max_threads=args.max_threads,
                discard_after=args.discard_after,
                scan_every=args.scan_every,
                silent=False if args.silent == 0 else True,
                model_dir=args.model_dir,
                detection_threshold=args.detection_threshold,
                max_feature_updates=args.max_feature_updates,
                static_model=True if args.no_3d_adapt == 1 else False,
                device=args.device,
                preproc_backend=args.preproc_backend
            )

        try:
            inference_start = time.perf_counter()
            faces = tracker.predict(frame)
            if len(faces) > 0:
                inference_time = (time.perf_counter() - inference_start)
                total_tracking_time += inference_time
                tracking_time += inference_time / len(faces)
                tracking_frames += 1

            packet = bytearray()
            detected = False

            for face_num, f in enumerate(faces):
                detected = True
                f = copy.copy(f)
                if f.eye_blink is None:
                    f.eye_blink = [1, 1]

                right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                left_state = "O" if f.eye_blink[1] > 0.30 else "-"

                if args.silent == 0:
                    print(f"Confidence[{f.id}]: {f.conf:.4f} / 3D fitting error: {f.pnp_error:.4f} / Eyes: {left_state}, {right_state}")
                if not f.success:
                    pts_3d = np.zeros((70, 3), np.float32)
                
                # Packet Block
                packet.extend(bytearray(struct.pack("d", now)))
                packet.extend(bytearray(struct.pack("i", f.id)))
                packet.extend(bytearray(struct.pack("f", width)))
                packet.extend(bytearray(struct.pack("f", height)))
                packet.extend(bytearray(struct.pack("f", f.eye_blink[0])))
                packet.extend(bytearray(struct.pack("f", f.eye_blink[1])))
                packet.extend(bytearray(struct.pack("B", 1 if f.success else 0)))
                packet.extend(bytearray(struct.pack("f", f.pnp_error)))
                packet.extend(bytearray(struct.pack("f", f.quaternion[0])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[1])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[2])))
                packet.extend(bytearray(struct.pack("f", f.quaternion[3])))
                packet.extend(bytearray(struct.pack("f", f.euler[0])))
                packet.extend(bytearray(struct.pack("f", f.euler[1])))
                packet.extend(bytearray(struct.pack("f", f.euler[2])))
                packet.extend(bytearray(struct.pack("f", f.translation[0])))
                packet.extend(bytearray(struct.pack("f", f.translation[1])))
                packet.extend(bytearray(struct.pack("f", f.translation[2])))
                for (x,y,c) in f.lms:
                    packet.extend(bytearray(struct.pack("f", c)))

                # Visualization
                if args.visualize > 1:
                    frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255))
                if args.visualize > 2:
                    frame = cv2.putText(frame, f"{f.conf:.4f}", (int(f.bbox[0] + 18), int(f.bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

                for pt_num, (x,y,c) in enumerate(f.lms):
                    packet.extend(bytearray(struct.pack("f", y)))
                    packet.extend(bytearray(struct.pack("f", x)))
                    if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.20):
                        continue
                    if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.20):
                        continue
                    x = int(x + 0.5)
                    y = int(y + 0.5)
                    if args.visualize != 0:
                        if args.visualize > 3:
                            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                        color = (0, 255, 0)
                        if pt_num >= 66:
                            color = (255, 255, 0)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            cv2.circle(frame, (y, x), 1, color, -1)
                if args.pnp_points != 0 and args.visualize != 0 and f.rotation is not None:
                    if args.pnp_points > 1:
                        projected = cv2.projectPoints(f.face_3d[0:66], f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    else:
                        projected = cv2.projectPoints(f.contour, f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                    for [(x,y)] in projected[0]:
                        x = int(x + 0.5)
                        y = int(y + 0.5)
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        y += 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                        x -= 1
                        if not (x < 0 or y < 0 or x >= height or y >= width):
                            frame[int(x), int(y)] = (0, 255, 255)
                for (x,y,z) in f.pts_3d:
                    packet.extend(bytearray(struct.pack("f", x)))
                    packet.extend(bytearray(struct.pack("f", -y)))
                    packet.extend(bytearray(struct.pack("f", -z)))
                if f.current_features is None:
                    f.current_features = {}
                for feature in features:
                    if not feature in f.current_features:
                        f.current_features[feature] = 0
                    packet.extend(bytearray(struct.pack("f", f.current_features[feature])))

            if detected and len(faces) < 40:
                sock.sendto(packet, (target_ip, target_port))

            if args.visualize != 0:
                cv2.imshow('OpenSeeFace Visualization', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            failures = 0
        except Exception as e:
            if e.__class__ == KeyboardInterrupt:
                if args.silent == 0:
                    print("Quitting")
                break
            traceback.print_exc()
            failures += 1
            if failures > 30:
                break

        del frame

        if target_duration > 0:
            duration = time.perf_counter() - frame_time
            if duration < target_duration:
                time.sleep(target_duration - duration)
        frame_time = time.perf_counter()

        if frame_time - last_gc_collect >= GC_INTERVAL_SECONDS:
            gc.collect()
            last_gc_collect = frame_time
except KeyboardInterrupt:
    if args.silent == 0:
        print("Quitting")

input_reader.close()
if args.visualize != 0:
    cv2.destroyAllWindows()

if args.silent == 0 and tracking_frames > 0:
    average_tracking_time = 1000 * tracking_time / tracking_frames
    print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
    print(f"Tracking time: {total_tracking_time:.3f} s\nFrames: {tracking_frames}")
