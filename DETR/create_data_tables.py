
from unidecode import unidecode
import easyocr
import numpy as np
import cv2
import os
import shutil
import csv

from PIL import Image, ImageEnhance

import pandas as pd
import re
import unicodedata  # Para normalizar texto y quitar acentos


# Función para quitar acentos y caracteres especiales
def quitar_acentos(texto):
    texto_normalizado = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto_limpio = re.sub(r'[^a-zA-Z0-9]', '_', texto_normalizado)
    texto_limpio = texto_limpio.replace('0', 'o')
    return texto_limpio


def pre_image(image_path):
  image = Image.open(image_path)

  image = image.convert("L")  # Convertir a escala de grises
  image = ImageEnhance.Contrast(image).enhance(50)
  image_np = np.array(image)
  mask_image_np = np.array(image)
  
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

def clear_tables(tables):
    
    format_tables = {
    'DATOS DEL PRODUCTO': 'A',
    'TITULAR': 'A',
    'FABRICANTE': 'A',
    'COMPOSICION': 'A',
    'ENVASES': 'A',
    'CONDICIONES GENERALES DE USO': 'A',
    'CLASE DE USUARIO': 'A',
    'MITIGACION DE RIESGOS EN LA MANIPULACION': 'A',
    'MITIGACION DE RIESGOS AMBIENTALES': 'A',
    'ELIMINACION DEL PRODUCTO Y/O CALDO': 'A',
    'GESTION DE ENVASES': 'A',
    'OTRAS INDICACIONES REGLAMENTARIAS': 'A',
    'CONDICIONES DE ALMACENAMIENTO': 'A',
    'USOS Y DOSIS AUTORIZADOS': 'B',
    'PLAZOS DE SEGURIDAD (PROTECCION DEL CONSUMIDOR)': 'C',
    'CLASIFICACIONES Y ETIQUETADO': 'D'
    }
    cleaned_tables = []
    
    
    for caption, image_path, bbox, name_directory in tables:
        if caption is not None and caption.strip() != 'None' and caption.strip() != '':
            if caption == "Clasificaclones y Etiquetado":
                caption = "CLASIFICACIONES Y ETIQUETADO"
            elif caption == "Eliminación del producto ylo caldo":
                caption = 'ELIMINACION DEL PRODUCTO Y/O CALDO'
            
            caption_clean = unidecode(caption).upper()
            tipo_tabla = format_tables.get(caption_clean, 'No definido')
            print(caption_clean, image_path, '----->', tipo_tabla)
            print('---'*50)
            if tipo_tabla != 'No definido':
                cleaned_tables.append({'caption': caption_clean, 'Imagen': image_path, 'Tipo': tipo_tabla, 'bbox': bbox, 'name_directory': name_directory})
                # cleaned_tables_dict[caption_clean] = {'Imagen': image_path, 'Tipo': tipo_tabla, 'bbox': bbox }
                # cleaned_tables.append(cleaned_tables_dict)
                # {'caption': caption_clean, 'Imagen': image_path, 'Tipo': tipo_tabla, 'bbox': bbox }
        
        
    
    asign_tables_type(cleaned_tables)
            # cleaned_tables.append(cleaned_tables_dict)
    # return cleaned_tables
    # asign_tables_type(cleaned_tables)
    
    
  
    
def asign_tables_type(cleaned_tables):
    
    # cleaned_tables = clear_tables(tables)
    
    # print(len(cleaned_tables))
    
    directory_csv = os.path.join('DETR', 'data_csv')

    #ELIMINO DATOS PREVIOS
    if os.path.exists(directory_csv):
        for entry in os.listdir(directory_csv):
            entry_path = os.path.join(directory_csv, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
    else:
         os.makedirs(directory_csv)
    
    #CONSTRUCTOR DE .CSV
    for table in cleaned_tables:
        caption = table['caption']
        image_path = table['Imagen']
        table_type = table['Tipo']
        bbox = table['bbox']
        name_directory = table['name_directory']
        bbox_filtered = [box for box in bbox if box['label'] != 'table column header']
        if table_type == 'A':
            ocr_tables_type_a([{'caption': caption, 'bbox':bbox_filtered, 'image': image_path, 'Tipo': table_type, 'name_directory': name_directory}])
        
#OCR para tablas tipo A
def ocr_tables_type_a(item_list_A):
    
    ruta_base = os.path.join('DETR', 'data_csv')
   
    data_type_A = []
    
    #CONSTRUIMOS .CSV
    for rows in item_list_A:
        caption = rows['caption']
        image_path = rows['image']
        image = pre_image(image_path)
        bbox = rows['bbox']
        name_directory = rows['name_directory']
        lines_to_write = []
        prompt_txt = {}
        
        for box in bbox:
            
            label = box['label']
            if label == 'table spanning cell' or label == 'table row':
                bbox_out = box['box']
                
                if label == 'DATOS DEL PRODUCTO':
                    cropped_image = image.crop((bbox_out['xmin'], bbox_out['ymin'], bbox_out['xmax'], bbox_out['ymax']))
                elif label == 'ENVASES':
                    cropped_image = image.crop((bbox_out['xmin'], bbox_out['ymin']-40, bbox_out['xmax'], bbox_out['ymax']+40))
                elif label == 'FABRICANTE':
                    cropped_image = image.crop((bbox_out['xmin'], bbox_out['ymin']-100, bbox_out['xmax'], bbox_out['ymax']+60))
                else:
                    cropped_image = image.crop((bbox_out['xmin'], bbox_out['ymin']-50, bbox_out['xmax'], bbox_out['ymax']))
                
                
                reader = easyocr.Reader(['en', 'es'])
                
                ocr_result = reader.readtext(np.array(cropped_image), detail=0, paragraph=True)
                row_text = " ".join(ocr_result)
                data_type_A.append({'caption': caption, 'content': row_text, 'image': image_path,'name_directory': name_directory})

        
                #creo los directorios por cada archivo para guardar .csv
                dir_path = os.path.join(ruta_base, name_directory)
                
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
            
                object_csv= [
                    ["caption", "content", "image_path", 'name_directory'],
                    [caption, row_text, image_path, name_directory]
                ]
                
                object_csv = [[x.replace(';', ',').replace('|', '/').replace('[', '') for x in fila] for fila in object_csv]
                
                #escribo el .csv para cada tabla detectada
                name_file = os.path.join(dir_path,  os.path.basename(image_path).split('.')[0] + '.csv')
                with open(name_file, mode='w', newline='', encoding='utf-8') as archivo_csv:
                    escritor_csv = csv.writer(archivo_csv, delimiter=',', quoting=1)
                    for fila in object_csv:
                        escritor_csv.writerow(fila)
                
                create_txt(dir_path, name_directory)
                
    
    return print('Fin creacion .csv')

def create_txt(dir_path, name_directory):
    
    txt_file_path = os.path.join(dir_path, 'file_txt.txt')

    
    open(txt_file_path, 'w').close()

    # Iterar sobre todos los archivos CSV en el directorio
    for csv_file in os.listdir(dir_path):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(dir_path, csv_file)
            with open(csv_path, mode='r', newline='', encoding='utf-8') as archivo_csv:
                lector_csv = csv.reader(archivo_csv)
                header = next(lector_csv)  # Leer la cabecera
                first_line = next(lector_csv, None)  # Leer la primera línea de datos
                if first_line:
                    line = first_line[1] + '\n'
                    with open(txt_file_path, mode='a', encoding='utf-8') as archivo_txt:
                        archivo_txt.write(f'{name_directory} | {line}')
    
    print('Fin creacion contexto')
        
