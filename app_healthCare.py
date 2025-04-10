import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Cargar el modelo
model = load_model("resnet_model.keras")

# Clases y descripciones
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
descriptions = {
    'Glioma': 'A type of tumor that starts in the glial cells of the brain or spine.',
    'Meningioma': 'A tumor that forms on membranes covering the brain and spinal cord.',
    'Pituitary': 'A tumor that occurs in the pituitary gland, affecting hormone production.',
    'No Tumor': 'No signs of a brain tumor were detected in the uploaded image.'
}

# Carpeta para imágenes subidas
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Crear app Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocesar imagen
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predicción
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][class_index]) * 100
            tumor_type = class_names[class_index]
            description = descriptions[tumor_type]

            return render_template("result.html",
                                   tumor_type=tumor_type,
                                   description=description,
                                   confidence=round(confidence, 2),
                                   image_path=filename)

    return render_template("index.html")

# Ruta para servir imágenes subidas
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ejecutar app
if __name__ == "__main__":
    app.run(debug=True)
