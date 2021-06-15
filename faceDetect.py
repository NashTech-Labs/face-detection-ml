import cv2 as cvobj


def run():
    input_image = cvobj.imread('image_add.jpg')
    grayscale_image = cvobj.cvtColor(input_image, cvobj.COLOR_BGR2GRAY)
    face_cascade = cvobj.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)

    for (column, row, width, height) in detected_faces:
        cvobj.rectangle(input_image, (column, row), (column + width, row + height), (0, 255, 0), 2)
    cvobj.imshow('Image', input_image)
    cvobj.waitKey(0)
    cvobj.destroyAllWindows()


if __name__ == '__main__':
    run()
