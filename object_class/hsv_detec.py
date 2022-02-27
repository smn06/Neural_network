import imutils
import cv2

colorRanges = [
	((29, 86, 6), (64, 255, 255), "green"),
	((57, 68, 0), (151, 255, 255), "blue")]

cam = cv2.VideoCapture(0)


while True:

	(grabbed, frame) = cam.read()
	frame = imutils.resize(frame, width=600)
	blur = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	for (lower, upper, colorName) in colorRanges:

		masking = cv2.inRange(hsv, lower, upper)
		masking = cv2.erode(masking, None, iterations=2)
		masking = cv2.dilate(masking, None, iterations=2)

		contour = cv2.findContours(masking.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		contour = imutils.grab_contours(contour)


		if len(contour) > 0:

			c = max(contour, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


			if radius > 10:
				cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.putText(frame, colorName, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
					1.0, (0, 255, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cam.release()
cv2.destroyAllWindows()