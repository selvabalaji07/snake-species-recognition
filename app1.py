from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
import torch
import io
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import os

app = Flask(__name__)

# Define the necessary variables
seed = 41
image_dim = 418
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Get the absolute path to the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model file
model_filename = "model.h5"
model_path = os.path.join(current_directory, model_filename)

# Load the trained model
nclass = 135  # Replace with the actual number of classes

# Assuming you have a model architecture defined (e.g., EfficientNet)
# Adjust the model architecture based on how you saved it
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=nclass)
model.fc = nn.Linear(model._fc.in_features, nclass)

# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(image_dim, image_dim)),
    transforms.Normalize((0.4718, 0.4429, 0.3738), (0.2519, 0.2388, 0.2393))
])

# Mapping between class indices and class names with venomous status
class_names = [
    {'name': 'Agkistrodon contortrix', 'venomous': True, 'country': 'United States', 'habitat': 'Forests, wetlands'},
    {'name': 'Agkistrodon piscivorus', 'venomous': True, 'country': 'United States', 'habitat': 'Swamps, marshes, wetlands'},
    {'name': 'Ahaetulla nasuta', 'venomous': False, 'country': 'India', 'habitat': 'Forests, trees'},
    {'name': 'Ahaetulla prasina', 'venomous': False, 'country': 'India', 'habitat': 'Trees, shrubs'},
    {'name': 'Arizona elegans', 'venomous': False, 'country': 'United States', 'habitat': 'Deserts, grasslands'},
    {'name': 'Aspidites melanocephalus', 'venomous': False, 'country': 'Australia', 'habitat': 'Deserts, scrublands'},
    {'name': 'Atractus crassicaudatus', 'venomous': False, 'country': 'South America', 'habitat': 'Forests, grasslands'},
    {'name': 'Austrelaps superbus', 'venomous': True, 'country': 'Australia', 'habitat': 'Forests, grasslands'},
    {'name': 'Bitis arietans', 'venomous': True, 'country': 'Africa', 'habitat': 'Savannas, grasslands'},
    {'name': 'Bitis gabonica', 'venomous': True, 'country': 'Africa', 'habitat': 'Rainforests, swamps'},
    {'name': 'Boa constrictor', 'venomous': False, 'country': 'South America', 'habitat': 'Forests, wetlands'},
    {'name': 'Bogertophis subocularis', 'venomous': False, 'country': 'United States', 'habitat': 'Forests, grasslands'},
    {'name': 'Boiga irregularis', 'venomous': False, 'country': 'Australia', 'habitat': 'Forests, coastal areas'},
    {'name': 'Boiga kraepelini', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, bushes'},
    {'name': 'Bothriechis schlegelii', 'venomous': True, 'country': 'Central America', 'habitat': 'Forests, mountains'},
    {'name': 'Bothrops asper', 'venomous': True, 'country': 'Central America', 'habitat': 'Forests, grasslands'},
    {'name': 'Bothrops atrox', 'venomous': True, 'country': 'South America', 'habitat': 'Rainforests, swamps'},
    {'name': 'Bungarus multicinctus', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, agricultural areas'},
    {'name': 'Carphophis amoenus', 'venomous': False, 'country': 'United States', 'habitat': 'Forests, grasslands'},
    {'name': 'Carphophis vermis', 'venomous': False, 'country': 'United States', 'habitat': 'Forests, grasslands'},
    {'name': 'Causus rhombeatus', 'venomous': True, 'country': 'Africa', 'habitat': 'Forests, savannas'},
    {'name': 'Cemophora coccinea', 'venomous': False, 'country': 'United States', 'habitat': 'Forests, grasslands'},
    {'name': 'Charina bottae', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, meadows'},
    {'name': 'Chrysopelea ornata', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, shrublands'},
    {'name': 'Clonophis kirtlandii', 'venomous': False, 'country': 'United States', 'habitat': 'Forests, grasslands'},
    {'name': 'Contia tenuis', 'venomous': False, 'country': 'United States', 'habitat': 'Forests, wetlands'},
    {'name': 'Corallus caninus', 'venomous': False, 'country': 'South America', 'habitat': 'Rainforests, swamps'},
    {'name': 'Corallus hortulanus', 'venomous': False, 'country': 'South America', 'habitat': 'Rainforests, swamps'},
    {'name': 'Coronella girondica', 'venomous': False, 'country': 'Europe', 'habitat': 'Forests, meadows'},
    {'name': 'Crotalus adamanteus', 'venomous': True, 'country': 'United States', 'habitat': 'Forests, grasslands'},
    {'name': 'Crotalus atrox', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, scrublands'},
    {'name': 'Crotalus cerastes', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, sand dunes'},
    {'name': 'Crotalus cerberus', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, rocky areas'},
    {'name': 'Crotalus lepidus', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, rocky areas'},
    {'name': 'Crotalus molossus', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, scrublands'},
    {'name': 'Crotalus ornatus', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, grasslands'},
    {'name': 'Crotalus ruber', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, scrublands'},
    {'name': 'Crotalus scutulatus', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, rocky areas'},
    {'name': 'Crotalus stephensi', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, scrublands'},
    {'name': 'Crotalus tigris', 'venomous': True, 'country': 'United States', 'habitat': 'Deserts, rocky areas'},
    {'name': 'Crotalus triseriatus', 'venomous': True, 'country': 'United States', 'habitat': 'Forests, mountains'},
	 {'name': 'Crotalus viridis', 'venomous': True, 'country': 'United States', 'habitat': 'Grasslands, deserts'},
    {'name': 'Crotaphopeltis hotamboeia', 'venomous': False, 'country': 'Africa', 'habitat': 'Forests, savannas'},
    {'name': 'Daboia russelii', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Dendrelaphis pictus', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, bushes'},
    {'name': 'Dendrelaphis punctulatus', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, shrublands'},
    {'name': 'Dendroaspis polylepis', 'venomous': True, 'country': 'Africa', 'habitat': 'Grasslands, savannas'},
    {'name': 'Diadophis punctatus', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Drymarchon couperi', 'venomous': False, 'country': 'United States', 'habitat': 'Forests, wetlands'},
    {'name': 'Elaphe dione', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Epicrates cenchria', 'venomous': False, 'country': 'South America', 'habitat': 'Rainforests, swamps'},
    {'name': 'Eunectes murinus', 'venomous': False, 'country': 'South America', 'habitat': 'Swamps, marshes'},
    {'name': 'Farancia abacura', 'venomous': False, 'country': 'United States', 'habitat': 'Wetlands, swamps'},
    {'name': 'Gonyosoma oxycephalum', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, agricultural areas'},
    {'name': 'Hemorrhois hippocrepis', 'venomous': False, 'country': 'Europe', 'habitat': 'Grasslands, scrublands'},
    {'name': 'Heterodon nasicus', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Heterodon simus', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Hierophis viridiflavus', 'venomous': False, 'country': 'Europe', 'habitat': 'Forests, meadows'},
    {'name': 'Hypsiglena torquata', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, scrublands'},
    {'name': 'Imantodes cenchoa', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, bushes'},
    {'name': 'Lampropeltis alterna', 'venomous': False, 'country': 'North America', 'habitat': 'Grasslands, rocky areas'},
    {'name': 'Lampropeltis calligaster', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Lampropeltis getula', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Lampropeltis pyromelana', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, rocky areas'},
    {'name': 'Lampropeltis triangulum', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Lampropeltis zonata', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Laticauda colubrina', 'venomous': True, 'country': 'Asia', 'habitat': 'Coastal areas, mangroves'},
    {'name': 'Leptodeira annulata', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, grasslands'},
    {'name': 'Leptophis ahaetulla', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, bushes'},
    {'name': 'Leptophis diplotropis', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, shrublands'},
    {'name': 'Leptophis mexicanus', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, grasslands'},
    {'name': 'Lycodon capucinus', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, agricultural areas'},
    {'name': 'Malpolon monspessulanus', 'venomous': True, 'country': 'Europe', 'habitat': 'Grasslands, scrublands'},
    {'name': 'Masticophis bilineatus', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, scrublands'},
    {'name': 'Masticophis lateralis', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, grasslands'},
    {'name': 'Masticophis schotti', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, scrublands'},
    {'name': 'Masticophis taeniatus', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, grasslands'},
    {'name': 'Micrurus fulvius', 'venomous': True, 'country': 'North America', 'habitat': 'Swamps, marshes'},
    {'name': 'Micrurus tener', 'venomous': True, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Morelia spilota', 'venomous': False, 'country': 'Australia', 'habitat': 'Forests, grasslands'},
    {'name': 'Morelia viridis', 'venomous': False, 'country': 'Australia', 'habitat': 'Rainforests, swamps'},
	{'name': 'Naja atra', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Naja naja', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Naja nivea', 'venomous': True, 'country': 'Africa', 'habitat': 'Forests, grasslands'},
    {'name': 'Natrix maura', 'venomous': False, 'country': 'Europe', 'habitat': 'Wetlands, marshes'},
    {'name': 'Nerodia cyclopion', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, marshes'},
    {'name': 'Nerodia floridana', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, swamps'},
    {'name': 'Nerodia taxispilota', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, marshes'},
    {'name': 'Ninia sebae', 'venomous': False, 'country': 'South America', 'habitat': 'Forests, grasslands'},
    {'name': 'Opheodrys aestivus', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, wetlands'},
    {'name': 'Ophiophagus hannah', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Oxybelis aeneus', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, bushes'},
    {'name': 'Oxyuranus scutellatus', 'venomous': True, 'country': 'Australia', 'habitat': 'Forests, grasslands'},
    {'name': 'Phyllorhynchus decurtatus', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Pituophis catenifer', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Pituophis deppei', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Protobothrops mucrosquamatus', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, mountains'},
    {'name': 'Psammodynastes pulverulentus', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Pseudaspis cana', 'venomous': False, 'country': 'Africa', 'habitat': 'Forests, grasslands'},
    {'name': 'Pseudechis australis', 'venomous': True, 'country': 'Australia', 'habitat': 'Forests, grasslands'},
    {'name': 'Pseudechis porphyriacus', 'venomous': True, 'country': 'Australia', 'habitat': 'Forests, grasslands'},
    {'name': 'Pseudonaja textilis', 'venomous': True, 'country': 'Australia', 'habitat': 'Forests, grasslands'},
    {'name': 'Python molurus', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Python regius', 'venomous': False, 'country': 'Africa', 'habitat': 'Forests, grasslands'},
    {'name': 'Regina septemvittata', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Rhabdophis subminiatus', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, wetlands'},
    {'name': 'Rhabdophis tigrinus', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, grasslands'},
    {'name': 'Rhadinaea flavilata', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, grasslands'},
    {'name': 'Rhinocheilus lecontei', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, grasslands'},
    {'name': 'Salvadora grahamiae', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, scrublands'},
    {'name': 'Salvadora hexalepis', 'venomous': False, 'country': 'North America', 'habitat': 'Deserts, scrublands'},
    {'name': 'Senticolis triaspis', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Sistrurus catenatus', 'venomous': True, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Sistrurus miliarius', 'venomous': True, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Spilotes pullatus', 'venomous': False, 'country': 'Central America', 'habitat': 'Forests, grasslands'},
    {'name': 'Tantilla coronata', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Tantilla gracilis', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Tantilla hobartsmithi', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Tantilla planiceps', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Thamnophis atratus', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, marshes'},
	{'name': 'Thamnophis couchii', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, marshes'},
    {'name': 'Thamnophis cyrtopsis', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, marshes'},
    {'name': 'Thamnophis marcianus', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, grasslands'},
    {'name': 'Thamnophis ordinoides', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, marshes'},
    {'name': 'Thamnophis proximus', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, grasslands'},
    {'name': 'Thamnophis radix', 'venomous': False, 'country': 'North America', 'habitat': 'Wetlands, grasslands'},
    {'name': 'Trimeresurus stejnegeri', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, mountains'},
    {'name': 'Tropidoclonion lineatum', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Tropidolaemus subannulatus', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, swamps'},
    {'name': 'Tropidolaemus wagleri', 'venomous': True, 'country': 'Asia', 'habitat': 'Forests, swamps'},
    {'name': 'Vipera ammodytes', 'venomous': True, 'country': 'Europe', 'habitat': 'Forests, rocky areas'},
    {'name': 'Vipera aspis', 'venomous': True, 'country': 'Europe', 'habitat': 'Forests, rocky areas'},
    {'name': 'Vipera seoanei', 'venomous': True, 'country': 'Europe', 'habitat': 'Forests, rocky areas'},
    {'name': 'Virginia valeriae', 'venomous': False, 'country': 'North America', 'habitat': 'Forests, grasslands'},
    {'name': 'Xenochrophis piscator', 'venomous': False, 'country': 'Asia', 'habitat': 'Forests, wetlands'}
]


def predict_snake(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        model.eval()
        output = model(image.to(device))
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        
        image_path = io.BytesIO(file.read())
        predicted_class = predict_snake(image_path)
        class_info = class_names[predicted_class]
        class_name = class_info['name']
        venomous = class_info['venomous']
        country = class_info['country']
        habitat = class_info['habitat']  # Fetch habitat information
        return jsonify({'class_name': class_name, 'venomous': venomous, 'country': country, 'habitat': habitat})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
