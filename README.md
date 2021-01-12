## Particularidades:

* *data_augmentation.py*: Funciona con el directorio `train` de tiny-imagenet-200. Se diseñó así debido a que en clase así se orientó. No obstante, se puede cambiar la lógica si se leen con `glob` y se transforma la ruta del fichero `x` en `x_augmented`. No se hizo ese cambio por cuestiones de tiempo.
* Los ejercicios:
    * *visual_tracking.py*
    * *eyefacedetector.py*
    * *pedestriandetector.py*
    * *text_recognition.py*
    * *visual_tracking_cv.py*
    
    Hacen uso de la función `proc_video` del paquete `utils` creada para el procesamiento de
  frames, ya sean provenientes de un `video` o `videosource` o una simple `imagen` . Aplica
  una función a cada uno de los frames. Concretamente la cabecera de la función es:
  
  ````python
  def video_transformation(video_source, output, fps, process_frame_function,
                         args=tuple(), window_name='Video', video_capture: cv2.VideoCapture = None):
    """
    Procesa un video desde una fuente de imagen y aplica la función "process_frame_function"
    a cada frame. Guarda los resultados en parámetro de salida "output"
    
    :param video_source: fuente de video entrante [int, path, image_seq]
    :param output: path de salida del resultado.
    :param fps: cantidad de frames del video de salida [int]
        este valor puede ser -1, 0, int+. 
        * Si es -1 el video de salida tendra DEFAULT_FPS fps.      
        * Si es 0 el valor de fps tomará el valor de la fuente de video.
        * Si <0, el video de salida tomará el valor valor especificado.
    :param process_frame_function: Función que es llamada en cada nuevo fotograma. Recibe siempre como primer
   parámetro el nuevo frame (BGR). Y como parámetros adicionales los especificados en "args". Esta función
   tiene que devolver un nuevo fotograma o bien el procesado o bien el mismo de entrada.
    
    :param args: Parámetros adicionales de la función "process_frame_function".
    :param window_name: Nombre de la ventana a mostrar. 
    :param video_capture: Este valor impide la creación de una fuente de video, sobreescribiéndolo por el 
  nuevo "video_capture".
    :return: None
    """
    pass
  ````
  
* *mediamatcher.py* Ofrece las combinaciones de matcher-detector siguientes:
    ````python
    MATCHER = {'bfm': cv2.BFMatcher, 'flann': cv2.FlannBasedMatcher}
    MATCHERS_DETECTORS_CONFIG = {
        "bfm": {
            'orb': {"normType": cv2.NORM_HAMMING, "crossCheck": True},
            'sift': {}
        },
        "flann": {
            'sift': {
                "indexParams": dict(algorithm=FLANN_INDEX_KD_TREE, trees=5),
                "searchParams": dict(checks=50)
            },
            'orb': {
                "indexParams": dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=12,  # 12
                    key_size=20,  # 20
                    multi_probe_level=1),
                "searchParams": {}
            }
        }
    }
    ````
  Para ejecutar con diferentes parámetros modificar los campos `indexParams` y `searchParams`. Puesto que 
en el lanzamiento del script solo se puede especificar cual usar pero no sus parámetros internos.
  
* *visual_tracking_cv.py* Ofrece los siguientes trackers:
    ````python
    TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    ````
  Se puede elegir cual usar en el lanzamiento del script, con el argumento `--tracker`.

* Se adjunta *Ejercicio10.py* y *Ejercicio10.ipynb*, donde en *Ejercicio10.ipynb* se ejecuta la función 
`search_candidates` del ejercicio *mediamatcher.py* de múltiples formas:
  * En secuencial, se ha ejecutado la función tal cual está implementada.
  * En varios hilos, utilizando `ThreadPoolExecutor` de la librería `concurrent.futures`.
  * En varios procesos, utilizando `ProcessPoolExecutor` de la librería `concurrent.futures`. Este se tiene
    que ejecutar en la consola debido a problemas con el `Jupyter`.
  * En varios clústeres, utilizando `IPyparallel`.
  
  En este enunciado se puede paralelizar en múltiples niveles. Ya que la búsqueda se hace con 1 imagen sobre
 un directorio, es conveniente paralelizar la búsqueda en los directorios (es lo que se realiza). No obstante,
  la función `search_candidates` puede ser paralilizada internamente ya que ella busca 1 imagen de entrada
  sobre un conjunto de imagenes también. Sin embargo, el rendimiento extra es inapreciable, por la limitación
  del número de núcleos.

        


  
