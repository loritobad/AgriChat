import os
from PIL import Image
from torchvision import transforms
import torch
from transformers import  AutoModelForObjectDetection
import shutil
from DETR.detect_structure import inference_table_structure


def load_images_to_table_detection():
    
    directory_img  = os.path.join("pdf2imgC", "img_pdf")
    # Lista para almacenar todas las rutas de las imágenes
    image_paths = []
    
    
    #leemos el nombre del directorio para no perder la referencia al contenido de cada pdf
    for filename in os.listdir(directory_img):
        item_path = os.path.join(directory_img, filename)
        for root, dirs, files in os.walk(item_path):
            for file_name in files:
                # Obtener la ruta completa del archivo
                file_path = os.path.join(root, file_name)
                # Verificar si el archivo es una imagen
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    # Agregar la ruta del archivo a la lista de rutas de imágenes
                    image_paths.append([filename, file_path])
    
    return image_paths

def procesing_detect_table():
    
     
     cropped_directory = os.path.join('DETR', 'cropped_img_tables')
    
     image_paths = load_images_to_table_detection()
     #print(image_paths)
     
     detection_table_model = AutoModelForObjectDetection.from_pretrained(os.path.join('models_download','detection_table', 'model'))
        
     # update id2label to include "no object"
     id2label = detection_table_model.config.id2label
     id2label[len(detection_table_model.config.id2label)] = "no object"
     
     tokens = []
     detection_class_thresholds = {
         "table": 0.9,
         "table rotated": 0.9,
         "no object": 1
         }
     
     #crop_padding = 0
     all_table_crops = []
     filenames = []
     for item in image_paths:
         filename = item[0]
         
         filenames.append(filename)
         
         
         image_path = item[1]
         
         
         image = Image.open(image_path).convert("RGB")
        
         
         pixel_values = detection_transform(image).unsqueeze(0)
       
         
         with torch.no_grad():
             outputs = detection_table_model(pixel_values)
        
     
         
         objects = outputs_to_objects(outputs, image.size, id2label)
         #print(objects)
         
         
         tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=2, filename = filename)
         print(f'Numero de tablas detectadas por imagen {len(tables_crops)}')
         
         all_table_crops.extend(tables_crops)
         print(f'Número de tablas TOTAL detectadas {len(all_table_crops)}')
         
         
         
     
     if os.path.exists(cropped_directory):
        for entry in os.listdir(cropped_directory):
            entry_path = os.path.join(cropped_directory, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
     else:
         os.makedirs(cropped_directory)
    
    
    #guardo los recortes de las tablas  
     guardar_table_crops(all_table_crops, cropped_directory, filenames)

     print('fin inferencia y recorte de tablas')
     print('llamando al detector de la estructura de las tablas')
     print('-'*50)
     inference_table_structure(filenames)
     print('-'*50)
         

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

#recortamos tablas detectadas
def objects_to_crops(img, tokens, objects, class_thresholds, filename, padding=15 ):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []

    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-10, bbox[2]-padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5] 
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
           
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        cropped_table = {'image': cropped_img, 'tokens': table_tokens, 'filename': filename}
        table_crops.append(cropped_table)

    return table_crops

#guardo los recortes

def guardar_table_crops(table_crops, directorio_guardado, filenames):
    
    for filename in filenames:
        subdirectory_path = os.path.join(directorio_guardado, filename)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)
        
        for i, table_crop in enumerate(table_crops): 
            if table_crop['filename'] == filename:
                nombre_archivo = f"tabla_{i}_{filename}.png"
                ruta_guardado = os.path.join(subdirectory_path,  nombre_archivo)
                table_crop['image'].save(ruta_guardado)

    