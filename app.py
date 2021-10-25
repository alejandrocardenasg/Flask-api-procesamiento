#LIBRERIAS
from flask import Flask, render_template, request, redirect
import argparse
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import os
import serial
import time
import threading
from google.cloud import storage
from datetime import datetime
import urllib.request
import random
import pose_estimation
import numpy as np
from numpy.lib.function_base import append
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


path_origin = os.path.join(os.path.dirname(__file__))
path_cred = os.path.join(path_origin,'tesismlac-4b9075ea4ca4.json')
path_file = os.path.join(path_origin,'file.json')
path_emgfiles = os.path.join(path_origin,'emgfiles')

#VARIABLES DE CREDENCIALES DE GOOGLE

cred = credentials.Certificate(path_cred)
BucketName = 'tesismlac.appspot.com'

# Conectar con la base de datos

try:
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    #Archivo de credenciales
    Credencial = storage.Client.from_service_account_json(json_credentials_path=path_cred)
    #Nombre Bucket
    bucket = Credencial.get_bucket(BucketName)
except:
    print("No se pudo conectar con la base de datos")

# INICIAR FLASK

app = Flask(__name__)

# VARIABLES DE FLASK 

app.config['UPLOAD_FOLDER'] = "archivo"

# RUTAS

@app.route('/')
def index():
    return "PROCESSING API REST"
	
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['imagen']
        id = request.form['post_id']
        dir = request.form['post_dir']
        #print(f.mimetype)
        #print(f.filename)
        filename = str(time.time()) + "_" +str(id)        
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename + ".jpg"))
        
        CloudFilename = str(id) + "/" + dir + "q/" + filename
        try:
            #Nombre archivo en la nube
            CloudName = bucket.blob(CloudFilename)
            #Dirección archivo local
            CloudName.upload_from_filename(("archivo/" + filename + ".jpg"), predefined_acl = 'publicRead')
            print("enviado a la nube")
        except:
            print("Error en Enviado a Cloud Storage")

        direccion = "archivo/" + filename + ".jpg"; 

        ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--mode", type=str, default="COCO",
                        help="Ingrese el modo")
        ap.add_argument("-i", "--img", type=str, default=direccion,
                        help="Ingrese la imagen de prueba")
        ap.add_argument("-s", "--verbose", type=int, default=0,
                        help="True/False mostrar datos")

        args = vars(ap.parse_args())
        mode = args["mode"]
        img_path = args["img"]
        verbose = args["verbose"]
        pose_obj = pose_estimation.PoseEstimation(mode)
        pose_obj.load_img(img_path)
        pose_obj.mainloop(verbose, filename)
        angulos = pose_obj.getAngles()

        try:
            CloudFilename = str(id) + "/" + dir + "q/" + filename + "-pro"
            #Nombre archivo en la nube
            CloudName = bucket.blob(CloudFilename)
            #Dirección archivo local
            CloudName.upload_from_filename(("resultado/" + filename + ".jpg"), predefined_acl = 'publicRead')
            print("enviado a la nube")
        except:
            print("Error en Enviado a Cloud Storage")

        doc_ref = db.collection('usuarios').document(str(id))
        doc_ref.update({
            u'filename': filename,
            u'angles_filename': angulos
        })

        os.remove('archivo/' + filename + '.jpg')
        os.remove('resultado/' + filename + '.jpg')

        url = 'http://' + request.remote_addr + ':3000/profile/watch/' + str(id) + "/" + dir
        print(url)
        return redirect(url)


@app.route('/emgp', methods = ['GET', 'POST'])
def emgp():

    promedio = []
    CONTADOR = 0
    if request.method == 'POST':
        id_Ref = request.form['id']
        while(CONTADOR < 6):
            try:
                blob_doc_ref = db.collection('files').document(str(id_Ref))
                blob_doc = blob_doc_ref.get()
                estado = blob_doc.to_dict()['estado']
                filename = blob_doc.to_dict()['filename']
                id = blob_doc.to_dict()['id']

                if(estado == True):
                    # OBTENER ARCHIVO DE CLOUD STORAGE
                    blob = bucket.blob(filename)
                    string = blob.download_as_text(encoding="utf-8")
                    jsonfile = json.loads(string)

                    #PROCESAMIENTO
                    try:
                        signal_emg = jsonfile['emg']
                        angx = jsonfile['angx']
                        angy = jsonfile['angy']
                        horas = jsonfile['horas']
                        nombre = jsonfile['nombre']
                        identificacion = jsonfile['identificacion']
                        emg = []
                        for x in signal_emg:
                            emg.append(float(x))


                        #plt.plot(cmv[:(len(cmv)//10)])

                        ##SACA SEÑAL ENVOLVENTE PARA ELIMINAR RUIDOS DE TRANSMISIÓN
                        q_u = np.zeros(np.shape(emg))
                        q_l = np.zeros(np.shape(emg))

                        u_x = [0,]
                        u_y = [emg[0],]

                        l_x = [0,]
                        l_y = [emg[0],]

                        for k in range(1,len(signal_emg)-1):
                            if (np.sign(emg[k]-emg[k-1])==1) and (np.sign(emg[k]-emg[k+1])==1):
                                u_x.append(k)
                                u_y.append(emg[k])

                            if (np.sign(emg[k]-emg[k-1])==-1) and ((np.sign(emg[k]-emg[k+1]))==-1):
                                l_x.append(k)
                                l_y.append(emg[k])

                        u_x.append(len(emg)-1)
                        u_y.append(emg[-1])

                        l_x.append(len(emg)-1)
                        l_y.append(emg[-1])

                        u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
                        l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)

                        for k in range(0,len(emg)):
                            q_u[k] = u_p(k)
                            q_l[k] = l_p(k)

                        
                        ###### CUADRO 1
                        cmv = []   ##SACA CMV DE LA SEÑAL
                        max_value = max(q_u)
                        cmv = list(map(lambda x: x / max_value, q_u))
                        cmv = np.array(cmv)

                        cmv_r = cmv - np.mean(cmv) ##ELIMINO OFFSET
                        #cmv_r = cmv_r.tolist()
                        for k in range(0,len(cmv_r)): #Convierto el emg a valor absoluto
                            if cmv_r[k] < 0:
                                cmv_r[k] = abs(cmv_r[k])*100 #Después de esto, la variable CMV_R es la señal de emg que se tiene que graficar de forma general

                        max_value = max(cmv_r)

                        peaks, _ = find_peaks(cmv_r,height = 0.3*max_value) 
                        peaks2, _ = find_peaks(cmv_r,height = 0.6*max_value) #Este es el valor de >60% de esfuerzo del CMV
                        peaks3, _ = find_peaks(cmv_r,height = (0.3*max_value, 0.6*max_value))#Este es el valor del 30% al 60% de esfuerzo del CMV
                        peaks4, _ = find_peaks(cmv_r, height = (0, 0.3*max_value))#Este es el valor de <30% de esfuerzo del CMV

                        p1 = len(peaks4)*0.02; p2 = len(peaks2)*0.02; p3 = len(peaks3)*0.02 #acá se imprimen todos los valores
                        esfuerzo = [p1,p2,p3]

                        #####CUADRO 2

                        tm = 0.2
                        #print(t)

                        peakst3, _ = find_peaks(cmv_r,width = (25*0.2), prominence=0.2*max_value) 
                        peakst32, _ = find_peaks(cmv_r,width = (25*0.2), prominence=0.7*max_value) 
                        #print(peakst3); 
                        est = len(peakst3)  # ESTO ES ACCIONES ESTÁTICAS DE ESFUERZO BAJO
                        est2 = len(peakst32) # ESTO ES ACCIONES ESTÁTICAS DE ESFUERZO ALTO
                        peakst4, _ = find_peaks(cmv_r,width = (tm,25*0.2),prominence=0.2*max_value)
                        peakst42, _ = find_peaks(cmv_r,width = (tm,25*0.2),prominence=0.7*max_value)
                        #print(peakst4); 
                        din = len(peakst4) # ESTO ES ACCIONES DINÁMICAS DE ESFUERZO BAJO
                        din2 = len(peakst42) # ESTO ES ACCIONES DINÁMICAS DE ESFUERZO ALTO

                        dtiem = (din*0.2) + (est*0.2) # ESTO ES TIEMPO TOTAL DE ESFUERZO BAJO
                        dtiem2 = (din2*0.2) + (est2*0.2) # ESTO ES TIEMPO TOTAL DE ESFUERZO ALTO

                        esfuerzo_bajo_alto = [
                            [din, din2],
                            [est,est2],
                            [round(dtiem,3),round(dtiem2,3)]
                        ]

                        ### CUADRO 3
                        lapso = cmv_r[:49]
                        lapso2 = cmv_r[49:99]
                        lapso3 = cmv_r[99:149]
                        lapso4 = cmv_r[149:199]
                        lapso5 = cmv_r[199:249]

                        peaks5, _ = find_peaks(lapso)
                        peaks52, _ = find_peaks(lapso2)
                        peaks53, _ = find_peaks(lapso3)
                        peaks54, _ = find_peaks(lapso4)
                        peaks55, _ = find_peaks(lapso5)

                        tab = len(peaks5)*5.0  #DATOS PROMEDIO ACCIONES X MIN 0 - 10
                        tab2 = len(peaks52)*5.0 #DATOS PROMEDIO ACCIONES X MIN 10 - 20
                        tab3 = len(peaks53)*5.0 #DATOS PROMEDIO ACCIONES X MIN 20 - 30
                        tab4 = len(peaks54)*5.0 #DATOS PROMEDIO ACCIONES X MIN 30 - 40
                        tab5 = len(peaks55)*5.0 #DATOS PROMEDIO ACCIONES X MIN 40 - 50

                        peaks6, _ = find_peaks(lapso, height=0.5*max_value)
                        peaks62, _ = find_peaks(lapso2, height=0.5*max_value)
                        peaks63, _ = find_peaks(lapso3, height=0.5*max_value)
                        peaks64, _ = find_peaks(lapso4, height=0.5*max_value)
                        peaks65, _ = find_peaks(lapso5, height=0.5*max_value)

                        ult = len(peaks6)*2  # DATOS PROMEDIO DE MAX ESFUERZOS 0 - 10 MIN
                        ult2 = len(peaks62)*0.2 # DATOS PROMEDIO DE MAX ESFUERZOS 10 - 20 MIN
                        ult3 = len(peaks63)*0.2 # DATOS PROMEDIO DE MAX ESFUERZOS 20 - 30 MIN
                        ult4 = len(peaks64)*0.2 # DATOS PROMEDIO DE MAX ESFUERZOS 30 - 40 MIN
                        ult5 = len(peaks65)*0.2 # DATOS PROMEDIO DE MAX ESFUERZOS 40 - 50 MIN

                        #PARA GRÁFICA DE EMG
                        cmv_ra = []
                        for i in cmv_r:
                            cmv_ra.append(i)
                        cmv_r = cmv_ra
                        
                        # PARA TABLA 3
                        promedio = [
                            [tab, ult],
                            [tab2, ult2],
                            [tab3, ult3],
                            [tab4, ult4],
                            [tab5, ult5]
                        ]

                    except:
                        print("Error al procesar los objetos")

                    print(promedio)

                    # CREAR ARCHIVO JSON
                    VALIDAR = False
                    Random = random.randint(0,10000)
                    entrada = os.path.join(path_emgfiles, str(time.time()) + "_" + str(Random) + ".json")
                    try:
                        data = {"nombre" : nombre,
                            "idenficacion":  identificacion,
                            "esfuerzo": esfuerzo,
                            "cmv_r": cmv_r,
                            "esfuerzo_bajo_alto": esfuerzo_bajo_alto,
                            "promedio": promedio
                        }
                        with open(entrada, 'w', encoding='utf8') as outfile:
                            json.dump(data, outfile, indent=4,ensure_ascii=False)
                        VALIDAR = True
                    except:
                        print("Error al crear el archivo local")
                    
                    print(promedio)

                    # SUBIR ARCHIVO A CLOUD STORAGE

                    if(os.path.isfile(entrada) and VALIDAR == True):
                        CloudFilename = filename + "-emg"
                        try:
                            #Nombre archivo en la nube
                            CloudName = bucket.blob(CloudFilename)
                            #Dirección archivo local
                            CloudName.upload_from_filename(entrada)
                            print("enviado a la nube")
                            os.remove(entrada)
                        except:
                            print("Error en Enviado a Cloud Storage")       

                        try:
                            doc_ref = db.collection('files').document(str(id))
                            query_ref = doc_ref.delete()
                            CONTADOR = CONTADOR + 1000
                        except:
                            print("Error al eliminar el registro de la base de datos")
                    

                    print(promedio)

                else:
                    CONTADOR = CONTADOR + 1000
                    print("El estado es falso")

            except:
                CONTADOR = CONTADOR + 1
                print("No se pudo acceder a la base de datos")
    print("Procesamiento exitoso")
    return "Procesamiento exitoso"
		
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000)