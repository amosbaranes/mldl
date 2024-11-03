
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo

# import dlib
# import cv2

class NAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 MLAlgo\n", dic, '\n', '-'*50)
        try:
            super(NAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 MAlgo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 MLAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class NDataProcessing(BaseDataProcessing, BasePotentialAlgo, NAlgo):
    def __init__(self, dic):
        # print("90567-010 MLDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 MLDataProcessing ", self.app)

    def cv2(self, dic):
        print("90122-cv2: \n", "="*50, "\n", dic, "\n", "="*50)
        # file_path = self.upload_file(dic)["file_path"]
        # print("file_path", file_path)
        # img = "/media" + file_path.split("media")[1]
        # s = img.split(".")
        # img_1 = s[0] + "_1." + s[1]
        #
        # s = file_path.split(".")
        # file_path_1 = s[0] + "_1." + s[1]
        #
        # try:
        #     image = cv2.imread(file_path)
        #     # Convert the image to grayscale (Dlib works with grayscale images)
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     detector = dlib.get_frontal_face_detector()
        #     faces = detector(gray)
        #     for face in faces:
        #         x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.imwrite(file_path_1, image)
        # except Exception as ex:
        #     print(str(ex))

        result = {"status": "ok cv2", "img": img, "img_1": img_1}
        return result



