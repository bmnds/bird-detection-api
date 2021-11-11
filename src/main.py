from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import flask
import io
import requests
import os.path
from functools import wraps

app = flask.Flask(__name__)
app.config['API_KEY'] = os.environ.get('API_KEY')
model = None
model_path = "../models/resnet50.model.bruno.h5"
classes = ['RED HEADED DUCK',
           'PINK ROBIN',
           'RED FACED CORMORANT',
           'BARRED PUFFBIRD',
           'WHIMBREL',
           'WALL CREAPER',
           'EVENING GROSBEAK',
           'MOURNING DOVE',
           'NORTHERN CARDINAL',
           'RED BEARDED BEE EATER',
           'RUDY KINGFISHER',
           'NORTHERN MOCKINGBIRD',
           'QUETZAL',
           'BLACK THROATED WARBLER',
           'BLACKBURNIAM WARBLER',
           'PUFFIN',
           'TURQUOISE MOTMOT',
           'NORTHERN FLICKER',
           'EASTERN BLUEBIRD',
           'SCARLET TANAGER',
           'CEDAR WAXWING',
           'CANARY',
           'AFRICAN CROWNED CRANE',
           'MAGPIE GOOSE',
           'VERMILION FLYCATHER',
           'ROBIN',
           'COMMON POORWILL',
           'BANDED BROADBILL',
           'HAWAIIAN GOOSE',
           'RED WINGED BLACKBIRD']

def token_required(f):
    '''
    Decorator to reject requests without the token
    '''
    @wraps(f)
    def decorator(*args, **kwargs):
        if 'X-API-KEY' not in flask.request.headers:
            return 'missing header X-API-KEY', 401
        
        token = flask.request.headers['X-API-KEY']
        if not token or token != app.config['API_KEY']:
            return 'invalid X-API-KEY', 401
        
        return f(*args, **kwargs)
    return decorator

def download_model():
    if not os.path.exists(model_path):
        print("baixando o modelo...")
        url = 'https://storage.googleapis.com/bird-detection-api/resnet50.model.bruno.h5'
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
        print("download concluído...")


def load_custom_model():
    global model
    model = load_model(model_path)


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return preprocess_input(image)


@app.route("/predict", methods=["POST"])
@token_required
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target=(244, 244))

            predictions = model.predict(image)[0]

            winner_bird, winner_proba = get_winning_class(predictions)
            data["bird"] = winner_bird
            data["probability"] = np.float64(winner_proba)

            data["predictions"] = to_dict(predictions)

            data["success"] = True

    return flask.jsonify(data)


def get_winning_class(preds):
    winner_idx = preds.argmax(axis=-1)
    return classes[winner_idx], preds[winner_idx]


def to_dict(preds):
    bird_dict = {}
    
    for bird, proba in zip(classes, preds):
        bird_dict[bird] = np.float64(proba)
    
    return bird_dict


# model exceeded 200mb and had to be hosted on google cloud storage
download_model()

load_custom_model()

# only starts the web server on main thread
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
