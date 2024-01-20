from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np

class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

model = models.load_model("baseline.keras")

port = 5001
logo_image = 'logo.png'
default_image = 'placeholder_image.png'
content = ''
prob = 0
pred = "" # predicted class

def predict_image(model, path_to_img):

    # normalize image
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32, 32))
    data = np.asarray(img) # turn it to tensor array
    data = data / 255
    probs = model.predict(np.array([data])[:1])

    top_prob = probs.max() # percentage that's the right for each classses 
    top_pred = class_names[np.argmax(probs)] # type of objects
    
    return top_prob, top_pred
    

index  = """
<|text-center
<|{logo_image}|image|width=50px|>

<|{content}|file_selector|extensions=.png|>
Select image

<|{pred}|>

<|text-center
<|{default_image}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
"""


def on_change(state, var_name, var_val):
    # catch content change
    if var_name == "content":
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = "this is a " + top_pred
        state.default_image = var_val

app = Gui(page=index)

if __name__ == "__main__":
    app.run(port=port, use_reloader=True)
