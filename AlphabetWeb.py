from flask import Flask
from gevent.pywsgi import WSGIServer
from flask import render_template
from flask import Response
import torch
import imutils
import time
import cv2
import cvzone
import warnings
from torch import nn
from torchvision import transforms, models
from PIL import Image as Imag
from copy import deepcopy
from cvzone.HandTrackingModule import HandDetector
warnings.filterwarnings("ignore", category=UserWarning) 

app = Flask(__name__)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "DenseNet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

def probabilidad(frame):
    img = frame
    img = imutils.resize(img, width=600)
    img = cv2.cvtColor(deepcopy(img), cv2.COLOR_BGR2RGB)
    image_pil = Imag.fromarray(img)
    input_size = 224
    data_transform = transforms.Compose([
      transforms.Resize(input_size),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = data_transform(image_pil)
    img = img.unsqueeze(0).to(device)
    out = model_ft(img).cpu().detach().tolist()[0]
    A_L, A_R, B_L, B_R, C_L, C_R, D_L, D_R, E_L, E_R, F_L, F_R, G_L, G_R, H_L, H_R, I_L, I_R, J_L, J_R, K_L, K_R, L_L, L_R, M_L, M_R, N_L, N_R, O_L, O_R, P_L, P_R, Q_L, Q_R, R_L, R_R, S_L, S_R, T_L, T_R, U_L, U_R, V_L, V_R, W_L, W_R, X_L, X_R, Y_L, Y_R, Z_L, Z_R = torch.nn.functional.softmax(torch.tensor(out)).tolist()
    symbols = {A_L:"A", A_R:"A", B_L:"B", B_R:"B", C_L:"C", C_R:"C", D_L:"D", D_R:"D", E_L:"E", E_R:"E", F_L:"F", F_R:"F", G_L:"G", G_R:"G", H_L:"H", H_R:"H", I_L:"I", I_R:"I", J_L:"J", J_R:"J", K_L:"K", K_R:"K", L_L:"L", L_R:"L", M_L:"M", M_R:"M", N_L:"N", N_R:"N", O_L:"O", O_R:"O", P_L:"P", P_R:"P", Q_L:"Q", Q_R:"Q", R_L:"R", R_R:"R", S_L:"S", S_R:"S", T_L:"T", T_R:"T", U_L:"U", U_R:"U", V_L:"V", V_R:"V", W_L:"W", W_R:"W", X_L:"X", X_R:"X", Y_L:"Y", Y_R:"Y", Z_L:"Z", Z_R:"Z"}
    maxim = max(torch.nn.functional.softmax(torch.tensor(out)).tolist())
    labels = str(symbols.get(maxim))
    return maxim, labels

model_ft, input_size = initialize_model("DenseNet", 52, False, use_pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft.eval()
model_ft.to(device)
model_ft.load_state_dict(torch.load('D:/best.pt')['model_state_dict'])

print("[INFO] starting video stream...")
#cap = cv2.VideoCapture('C:/video_data/K_R.mp4') #J
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2.0)

detector = HandDetector(detectionCon = 0.8, maxHands = 1)

def generate():
    while True:
        success, img = cap.read()
        hands = detector.findHands(img, draw=False)
        
        if hands:
            hand1 = hands[0]
            if hand1 and "bbox" in hand1[0]:
                bbox = hand1[0]['bbox']
                x, y, ancho, alto = bbox
                x1, y1, x2, y2 = x, y, ancho, alto
                
                x1, y1 = max(x1,0), max(y1,0)
                
                img = cvzone.cornerRect(img, (x1-20, y1-20, x2+40, y2+40))
                
                maximo, etiqueta = probabilidad(img)
                maximo = str(maximo*100)+str('%')
                
                t1 = x1 - 25, y1 - 70
                t2 = x1 - 25, y1 - 40
                
                cv2.putText(img, etiqueta, (int(t1[0]), int(t1[1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                cv2.putText(img, maximo, (int(t2[0]), int(t2[1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                
                (flag, encodedImage) = cv2.imencode(".jpg", img)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
@app.route("/")
def index():
     return render_template("index.html")
@app.route("/video_feed")
def video_feed():
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")
if __name__ == "__main__":
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    app.run(debug=False)
cap.release()
