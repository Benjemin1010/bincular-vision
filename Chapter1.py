import cv2
print("Package Imported")
# img=cv2.imread("resources/cali.jpg")
# cv2.imshow("output",img)
# cv2.waitKey(0)
cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

while True:
    flag, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



