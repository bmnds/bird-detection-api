# import the necessary packages
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
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


def load_custom_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model('../models/resnet50.model.bruno.h5')


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(244, 244))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)

            bird, proba = get_class(preds[0])
            data["bird"] = bird
            data["probability"] = np.float64(proba)

            probas = pretty_print_preds(preds[0])
            data["predictions"] = probas

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


def get_class(preds):
    i = preds.argmax(axis=-1)
    bird, prob = classes[i], preds[i]
    print(f"{bird} {prob*100:.2f}%")
    return bird, prob


def pretty_print_preds(preds):
    bird_dict = {}
    for bird, proba in zip(classes, preds):
        bird_dict[bird] = np.float64(proba)
    print(f"{bird_dict}")
    return bird_dict


load_custom_model()

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
