from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import base64
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

labels = ['hippopotamus', 'sparrow', 'gorilla', 'cat', 'rhinoceros', 'wombat', 'seahorse', 'butterfly', 'donkey', 'raccoon', 'dragonfly', 'crab', 'pig', 'orangutan', 'turtle', 'antelope', 'dog', 'bee', 'coyote', 'fox', 'pigeon', 'dolphin', 'fly', 'turkey', 'boar', 'goldfish', 'hare', 'bear', 'penguin', 'squid', 'zebra', 'leopard', 'sheep', 'hamster', 'panda', 'mosquito', 'lobster', 'duck', 'ox', 'owl', 'tiger', 'whale', 'crow', 'rat', 'moth', 'eagle', 'reindeer', 'grasshopper', 'otter', 'starfish', 'hyena', 'goat', 'sandpiper', 'seal', 'jellyfish', 'hummingbird', 'mouse', 'hornbill', 'porcupine', 'wolf', 'lizard', 'woodpecker', 'beetle', 'chimpanzee', 'parrot', 'kangaroo', 'pelecaniformes', 'oyster', 'caterpillar', 'okapi', 'ladybugs', 'bat', 'cockroach', 'koala', 'swan', 'octopus', 'hedgehog', 'horse', 'flamingo', 'squirrel', 'bison', 'cow', 'deer', 'lion', 'goose', 'shark', 'snake', 'badger', 'elephant', 'possum']
labels_kor = ['하마', '참새', '고릴라', '고양이', '코뿔소', '웜뱃', '해마', '나비', '당나귀', '너구리', '잠자리', '게', '돼지', '오랑우탄', '거북이', '영양', '개', '벌', '코요테', '여우', '비둘기', '돌고래', '파리', '칠면조', '멧돼지', '금붕어', '토끼', '곰', '펭귄', '오징어', '얼룩말', '표범', '양', '햄스터', '팬더', '모기', '랍스터', '오리', '소', '올빼미', '호랑이', '고래', '까마귀', '쥐', '나방', '독수리', '순록', '메뚜기', '수달', '불가사리', '하이에나', '염소', '샌드파이퍼', '물개', '해파리', '벌새', '쥐', '코뿔소', '고슴도치', '늑대', '도마뱀', '딱따구리', '딱정벌레', '침팬지', '앵무새', '캥거루', '펠레카니목', '굴', '애벌레', '오카피', '무당벌레', '박쥐', '바퀴벌레', '코알라', '백조', '낙지', '고슴도치', '말', '플라밍고', '다람쥐', '들소', '소', '사슴', '사자', '거위', '상어', '뱀', '오소리', '코끼리', '주머니쥐']

model = tf.keras.models.load_model("./model/animal_classfication_xception_model.h5")

print("########## 분류 가능한 동물 목록(총 90가지) *벌레 주의* ##########")
print(labels_kor)

@api_view(["POST"])
def predict(request):
    base64_string = request.data.get('image')
    img = Image.open(BytesIO(base64.b64decode(base64_string)))
    
    img = np.array(img)
    resized_img = cv2.resize(img,(224,224))
    resized_img = resized_img / 255.0
    resized_img = np.expand_dims(resized_img, axis=0)
    
    rt = model.predict(resized_img)
    rt = np.argmax(rt, axis = 1)
    
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    
    idx = labels.index(le.inverse_transform([rt])[0])
    rt_dict = {'result' : le.inverse_transform([rt])[0], 'kor_result': labels_kor[idx]}
    
    print(rt_dict['result'])
    print(labels_kor[idx])
    return Response(rt_dict)