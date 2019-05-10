# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : yoloface.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import sys
import os

from yoloface.utils import *

#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default=yolov3_cfg_path,
                    help='path to config file')
parser.add_argument('--model-weights', type=str, default=yolov3_model_weights_path,
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='', nargs='+',
                    help='path to image file(s)')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
parser.add_argument('--gui', action='store_true',
                    help='Show GUI')
args = parser.parse_args()

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to image file: ', args.image)
print('[i] Path to video file: ', args.video)
print('###########################################################\n')

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

def _main():

    detector = yolo_dnn_face_detection_model_v3(args.model_cfg, args.model_weights)

    if args.gui:
        wind_name = 'face detection using YOLOv3'
        cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''

    if args.image:
        # validate images
        for img in args.image:
            if not os.path.isfile(img):
                print("[!] ==> Input image file {} doesn't exist".format(img))
                args.image.remove(img)

        if len(args.image) == 0:
            print("[!] ==> No images to detect faces on")
            sys.exit(1)

        cap = ImageVideoCapture(args.image)
        output_files = [img[:-4].rsplit('/')[-1] + '_yoloface.jpg' for img in args.image]
    elif args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Input video file {} doesn't exist".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(args.src)

    # Get the video writer initialized to save the output video
    if not args.image:
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       cap.get(cv2.CAP_PROP_FPS), (
                                           round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            if not args.image:
                print('[i] ==> Output file is stored at {}'.format(os.path.join(args.output_dir, output_file)))
            cv2.waitKey(1000)
            break

        if output_files: 
            output_file = output_files.pop(0)

        boxes = detector(frame)

        print('[i] ==> # detected faces: {}'.format(len(boxes)))

        for i, box in enumerate(boxes):
            left, top, width, height, confidence = box

            #draw_predict(frame, confidence, left, top, left + width, top + height)
            left, top, right, bottom = refined_box(left, top, width, height)
            draw_predict(frame, confidence, left, top, right, bottom)

        if args.gui:
            # initialize the set of information we'll displaying on the frame
            info = [
                ('number of faces detected', '{}'.format(len(faces)))
            ]

            for (i, (txt, val)) in enumerate(info):
                text = '{}: {}'.format(txt, val)
                cv2.putText(frame, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Save the output video to file
        if args.image:
            cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
            print('[i] ==> Output file is stored at {}'.format(os.path.join(args.output_dir, output_file)))
        else:
            video_writer.write(frame.astype(np.uint8))

        if args.gui:
            cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
