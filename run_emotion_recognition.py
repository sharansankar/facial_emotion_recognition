import cv2
import numpy as np
import keras
import numpy as np
import argparse
from keras.models import load_model

classes = ["anger","disgust","fear","happy","sad","surprise","neutral"]


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Performing facial emotion recognition.')
    parser.add_argument("-p","--network_path",type=str, default="keras_models/facial_cnn.h5" )
    args = parser.parse_args()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    video_capture = cv2.VideoCapture(0)

    nn = load_model(args.network_path)

    while True:
        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print("gray shape: ", np.shape(gray))
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            tl_x = int(1.07*x)
            tl_y = int(1.17*y)
            br_x = int(x+0.90*w)
            br_y = int(y+0.95*h)


            # try:
            #get face ROI
            face_roi = gray[tl_y:br_y,tl_x:br_x]
            # print(face_roi.mean())
            # face_roi = face_roi - face_roi.mean()
            face_roi = np.multiply(face_roi,1/255.0)
            tmp = cv2.resize(face_roi, (48,48))
            face_roi = cv2.resize(face_roi, (48,48))
            face_roi = face_roi.reshape(1,48,48,1)
            prediction = np.argmax(nn.predict(face_roi,batch_size=1))
            # except:
            #     print "failed"
            cv2.rectangle(frame, (tl_x,tl_y), (br_x, br_y), (0, 255, 0), 2)
            cv2.putText(frame,"emotion: {}".format(classes[prediction]), (tl_x,tl_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))


        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
