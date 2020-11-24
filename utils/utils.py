import cv2
from tqdm import tqdm

OUT_CODEC = 'MJPG'
DEFAULT_FPS = 10


def video_transformation(video_source, output, fps, process_frame_function, args=tuple()):
    """
    Process a video from a video source and apply the callable process_frame_function
    in every frame. Save the result in the output param.

    :param video_source: video source input [int, path, image_seq]
    :param output: output to save the processed video
    :param fps: the amount of frames of the output video [:int]
        this value can be -1, 0, int +. If -1 then the result video will have the
        DEFAULT_FPS amount. If 0 this reads the video source and take the value
        from there. Else this take any positive value.
    :param process_frame_function: function called in every video frame, this receives
    the frame (BGR) as the first parameter and args with rest of the parameters. This
    function must return the processed/transformed or non-modified frame.
    :param args: optional parameters

    :return: None
    """
    # Open the video
    video = cv2.VideoCapture(video_source)
    # Create the codec to save the video
    fourcc = cv2.VideoWriter_fourcc(*OUT_CODEC)

    # Create the video properties
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

    if fps == -1:
        v_fps = DEFAULT_FPS
    elif fps == 0:
        v_fps = video.get(cv2.CAP_PROP_FPS)
    else:
        v_fps = fps

    amount_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Create the video writer
    video_writer = cv2.VideoWriter(
        output, fourcc, v_fps, (int(width), int(height))
    )

    progress_bar = tqdm(total=amount_frames)
    while video.isOpened():
        ret, frame = video.read()  # Get frame
        if ret:
            processed_frame = process_frame_function(frame, *args)  # Process the frame.
            cv2.imshow("Video", processed_frame)  # Show the video
            video_writer.write(processed_frame)  # Write frame
            progress_bar.update(1)
        else:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()
