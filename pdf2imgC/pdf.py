from pdf2image import convert_from_bytes
import os
import shutil

def convert_pdf_to_img(pdf_bytes, file_name):
    
    file_name = file_name.split('.')[0]
    output_directory = os.path.join("pdf2imgC", "img_pdf", file_name)

    # Crear el directorio de salida si no existe
    os.makedirs(output_directory, exist_ok=True)
     
     #convertimos paginas de pdf a imagen
    images = convert_from_bytes(pdf_file= pdf_bytes )
          
    #guardamos las imagenes escaneadas del pdf
    for i, image in enumerate(images):
        image_path = os.path.join(output_directory, f"page_{i+1}_{file_name.split('.')[0]}.jpg")
        image.save(image_path, "JPEG")

def clear_directory():
    
    delete_directory = os.path.join("pdf2imgC", "img_pdf")
    #borro todo el contenido previo
    for item in os.listdir(delete_directory):
        item_path = os.path.join(delete_directory, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Elimina el directorio y todo su contenido
            elif os.path.isfile(item_path):
                os.remove(item_path)  # Elimina el archivo
        except Exception as e:
            print(f"No se pudo eliminar {item_path}: {e}")