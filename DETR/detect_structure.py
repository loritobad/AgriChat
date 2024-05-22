import os
import cv2
import numpy as np
import torch
import easyocr

from DETR.create_data_tables import asign_tables_type,clear_tables

from transformers import  AutoModelForObjectDetection
from PIL import Image, ImageEnhance
from torchvision import transforms

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image


def inference_table_structure(filenames):
    
     #leemos tablas recortadas con su path
    directory = os.path.join('DETR', 'cropped_img_tables')
    image_paths = []
    
    for filename_dir in list(set(filenames)):
        subdirectory_path = os.path.join(directory, filename_dir)
        elementos = os.listdir(directory)
        for item in elementos:
            if item == filename_dir:
                file = os.listdir(subdirectory_path)
                for name in file:
                    file_path = os.path.join(subdirectory_path, name)
                # print(subdirectory_path, item, image_paths)
                    image_paths.append([file_path, item])

    #cargamos el modelo
    detection_structure_model = AutoModelForObjectDetection.from_pretrained(os.path.join('models_download','detection_structure', 'model'))
    
    detection_transform = transforms.Compose([
        MaxResize(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        ])
    
    id2label = detection_structure_model.config.id2label
    id2label[len(detection_structure_model.config.id2label)] = "no object"
    
    captions = []
    reader = easyocr.Reader(['en', 'es'])
    
    #Inferencia sobre cada image
    for item in image_paths:
         image_path = item[0]
         name_directory = item[1]
        
        
         
         #preprocesamos la imagen
        #  image = pre_image(image_path)
         image = Image.open(image_path)
         print('IMAGEN preprocesada')
         
         #normalizamos y pasamos a tensor
         pixel_values = detection_transform(image).unsqueeze(0)
         
       
         #prediccion
         with torch.no_grad():
             output = detection_structure_model(pixel_values)
             
         objects = outputs_to_objects(output, image.size, id2label)
         #ordeno los bbox por fila de cada imagen
         bounding_boxes = extract_and_sort_bounding_boxes(objects, padding=5)
         print('Bbox ordenado por fila terminado')
         
         #construyo una lista con los nombres de cabeza de tablas para poder filtrar
         caption = detect_caption( bounding_boxes, image_path, 'table column header', reader)
         captions.append([caption, image_path, bounding_boxes, name_directory])
    
    clear_tables(captions)
    
   
    
    
    


def procesing_detect_structure():
    pass

def pre_image(image_path):
  image = Image.open(image_path)

  image = image.convert("L")  # Convertir a escala de grises
  image = ImageEnhance.Contrast(image).enhance(50)
  image_np = np.array(image)
  
  
  # Aplica una operación de apertura morfológica para eliminar los puntos negros
  kernel = np.ones((1, 1), np.uint8)
  image_np = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)
  image_np = cv2.GaussianBlur(image_np, (5,5), 0)  # Desenfoque
#   _, image_np = cv2.threshold(image_np, 120, 255, cv2.THRESH_BINARY)

#     # Se puede cambiar el tamaño para ajustar el nivel de erosión
  elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Tamaño de 3x3

#     # Aplicar erosión para reducir el grosor de los caracteres
  image_np = cv2.erode(image_np, elemento_estructurante, iterations=3)
  
  
  image = Image.fromarray(image_np)

  return image

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

#ordena los bbox por fila de cada imagen
def extract_and_sort_bounding_boxes(results, padding=0):
    bounding_boxes = []

    for result in results:
        bbox = result['bbox']
        # Aplica el padding y evita coordenadas negativas
        xmin = max(0, bbox[0] - padding)
        ymin = max(0, bbox[1] - padding)
        xmax = bbox[2] + padding
        ymax = bbox[3] + padding

        bounding_box = {
            'box': {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            },
            'label': result['label'],
            'score': result['score']  # Mantiene el score para referencia
        }
        bounding_boxes.append(bounding_box)

    # Ordena por coordenada Y para mantener el orden correcto
    bounding_boxes.sort(key=lambda x: x['box']['ymin'])

    return bounding_boxes

def detectar_fila_con_label(bounding_boxes, label):
    
    for box in bounding_boxes:
        if box['label'] == label:
            bounding_box_result = { 'box': {
                                            'xmin': box['box']['xmin'],
                                            'ymin': box['box']['ymin'], 
                                            'xmax': box['box']['xmax'],
                                            'ymax': box['box']['ymax']
                                            }
                                   }
            return  bounding_box_result
        
      

# Detecta los nombres de las tablas
def detect_caption(bounding_boxes, image_path, label, reader):
    # Detectar la fila con el label especificado
    caption = detectar_fila_con_label(bounding_boxes, label)
   
    if caption:
          image = pre_image(image_path)
        #   image = Image.open(image_path)
          bbox = caption['box']
          bbox = add_padding(bbox, 7, -1)
          cropped_image = image.crop((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
         
    
        #debug
        #   cabecera = os.path.join('DETR', 'img_recortadas', os.path.basename(image_path))
        #   image_np = np.array(cropped_image)
        #   cv2.imwrite(cabecera,image_np )
          
          ocr_result = reader.readtext(np.array(cropped_image), detail=0)
          ocr_text = " ".join(ocr_result)
          return str(ocr_text)
    else:
        return caption
  
def add_padding(bbox, padding_top, padding_low):
    # Ajustar la coordenada ymin con el padding
    new_ymax = max(bbox['ymax'] - padding_low, 0)
    new_ymin = bbox['ymin'] - padding_top
    #print(f'viejo: {bbox} ------ nuevo: {new_ymax}')
    
    return {
        'xmin': bbox['xmin'],  
        'ymin': new_ymin,
        'xmax': bbox['xmax'],  
        'ymax': new_ymax
    }
   
   