import cv2
import os


video_path = r"data\processed\darkact\Thermal\sit\sit_250621234150_00018_inf.mp4"


print("=" * 60)
print("THERMAL VIDEO CHECK")
print("=" * 60)


print("\nFile exists:")
print(os.path.exists(video_path))


cap = cv2.VideoCapture(video_path)


print("\nVideo opened:")
print(cap.isOpened())


success, frame = cap.read()


if success:

    print("\nFrame shape:")
    print(frame.shape)

    print("\nData type:")
    print(frame.dtype)

    print("\nMinimum:")
    print(frame.min())

    print("\nMaximum:")
    print(frame.max())

    print("\nMean:")
    print(frame.mean())

else:

    print("\nCould not read frame.")


cap.release()


print("\n" + "=" * 60)
print("THERMAL CHECK COMPLETED")
print("=" * 60)