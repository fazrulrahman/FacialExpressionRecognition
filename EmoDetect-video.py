import argparse,sys
sys.path.insert(0,'/usr/local/lib/python2.7/site-packages')

try:
    from FeatureGen import*
except ImportError:
    print "Make sure FeatureGen.pyc file is in the current directory"
    exit()

try:
    import dlib
    from skimage import io
    import numpy
    import cv2
    from sklearn.externals import joblib
except ImportError:
        print "Make sure you have OpenCV, dLib, scikit learn and skimage libraries properly installed"
        exit()

emotions={ 1:"Anger", 2:"Contempt", 3:"Disgust", 4:"Fear", 5:"Happy", 6:"Sadness", 7:"Surprise"}

webcamvid = cv2.VideoCapture(0)

def Predict_Emotion():
    while 1:
        ret, img = webcamvid.read()
    
        win.clear_overlay()
        win.set_image(img)

        dets=detector(img,1)

        for k,d in enumerate(dets):

                shape=predictor(img,d)
                landmarks=[]
                for i in range(68):
                    landmarks.append(shape.part(i).x)
                    landmarks.append(shape.part(i).y)
        
    
                landmarks=numpy.array(landmarks)
    
                print "Generating features......"
                features=generateFeatures(landmarks)
                features= numpy.asarray(features)

                print "Performing PCA Transform......."
                features1=[[features][0]]
                pca_features=pca.transform(features1)

                print "Predicting using trained model........"
                emo_predicts=classify.predict(pca_features)
                print "Predicted emotion using trained data is { " + emotions[int(emo_predicts[0])] + " }"
                print ""

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,emotions[int(emo_predicts[0])],(20,20), font, 1,(0,255,255),2)

                win.add_overlay(shape)

        cv2.namedWindow("Output")
        cv2.imshow("Output",img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


if __name__ == "__main__":


    landmark_path="shape_predictor_68_face_landmarks.dat"

    print "Initializing Dlib face Detector.."
    detector= dlib.get_frontal_face_detector()

    print "Loading landmark identification data..."
    try:
        predictor= dlib.shape_predictor(landmark_path)
    except:
        print "Unable to find trained facial shape predictor. \nYou can download a trained facial shape predictor from: \nhttp://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2"
        exit()

    win=dlib.image_window()

    print "Loading trained data....."

    try:
        classify=joblib.load("traindata.pkl")
        pca=joblib.load("pcadata.pkl")
    except:
        print "Unable to load trained data. \nMake sure that traindata.pkl and pcadata.pkl are in the current directory"
        exit()

    Predict_Emotion()


