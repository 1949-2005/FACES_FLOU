import os
import argparse

import cv2
import mediapipe as mp


def process_img(img, face_detection):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default='video')
args.add_argument("--filePath", default='./data/video.mp4')

args = args.parse_args()


# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filePath)

        img_processed = process_img(img, face_detection)

        # Display the processed image
        cv2.imshow("video", img_processed)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        # Prepare output video writer
        output_video = cv2.VideoWriter('./output/output.mp4',
                                        cv2.VideoWriter_fourcc(*'MP4V'),
                                        int(cap.get(cv2.CAP_PROP_FPS)),
                                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while ret:

            frame_processed = process_img(frame, face_detection)

            # Display the processed frame
            cv2.imshow("Processed Frame", frame_processed)

            # Write processed frame to output video
            output_video.write(frame_processed)

            # Introduce a delay to control the frame rate
            delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))  # Calculate delay based on video FPS
            if delay < 1:
                delay = 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):  # Press 'q' to quit
                break

            ret, frame = cap.read()

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
