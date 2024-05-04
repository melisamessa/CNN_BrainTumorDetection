<h1>Red convolucional de clasificación binaria para la detección de tumores cerebrales</h1>
<h3>Repositorio creado para la signatura Neuroscience of Learning Machines en la carrera de Ingeniería de Software de la Universidad Nacional del Centro de la Provincia de Buenos Aires de Tandil.</h3>
<h3>Fecha de realización: Septiembre 2023</h3>

<h4>1. Introducción</h4>
<p>Los tumores cerebrales, ya sean benignos o malignos, pueden tener un impacto
significativo en la salud y la calidad de vida de los pacientes si no se identifican y se
tratan de manera eficiente. Las imágenes de resonancia magnética se han
convertido en una herramienta fundamental para la detección y el diagnóstico de
estos tumores cerebrales.</p>

<p>En este contexto, el presente proyecto se centra en la implementación y evaluación
de una Red Convolucional de Clasificación Binaria diseñada para abordar el desafío
de la detección de tumores cerebrales en imágenes de resonancia magnética. La
detección automatizada de tumores cerebrales a partir de imágenes desempeña un
papel importante en la atención médica moderna.</p>

<p>El objetivo principal del proyecto fue desarrollar una CNN capaz de clasificar
imágenes de resonancia magnética en dos categorías: “Normal” y “Anómala”. Está
clasificación se basó en la capacidad de la red para aprender de manera automática
las características importantes de las imágenes que distinguen entre el tejido
cerebral sano y el enfermo.</p>

<p>Además se abordó la exploración de autoencoders para la tarea de la detección de
anomalías en imágenes médicas comparando su rendimiento con la arquitectura
elegida, ya que se optó por un enfoque de clasificación binaria gracias a las
capacidades de las CNN para aprender características discriminativas específicas
de cada clase.</p>

<h4>2. Dataset</h4>
<p>El dataset elegido fue provisto por Kaggle. Este se compone de imágenes
de resonancia magnética de cerebros separados en dos carpetas: cerebros normal
y cerebros anómalos.</p>

<p>Un cerebro normal para el presente proyecto, es aquel que no posee ningún tumor
cerebral. Y un cerebro anómalo es aquel que posee al menos un tumor ya sea
benigno o maligno. Esta primera instancia del trabajo no distingue entre los
diferentes tipos de tumores, si no que se centra únicamente en la detección de la
anomalía que en este caso sería encontrar algún tipo de tumor.</p>

<h4>3. Herramientas utilizadas</h4>
<p><strong>Tensorflow</strong> es una librería de código libre para computación numérica usando grafos de flujo de datos, 
  es utilizada para trabajar en Inteligencia Artificial.</p>
<p><strong>Keras</strong> es una API de alto nivel para la construcción y el entrenamiento de modelos
de aprendizaje profundo que se encuentra integrada en TensorFlow.</p>
<p>Ambas herramientas antes mencionadas, fueron utilizadas para definir la
arquitectura de la red neuronal, compilarla y entrenarla.</p>

<p><strong>ImageDataGenerator</strong> es una clase de Keras útil para trabajar con cantidades
grandes de imágenes ya que permite cargar, procesar y aumentar imágenes en
tiempo real mientras se entrena el modelo. Esto ahorra recursos como la memoria
teniendo en cuenta que el presente trabajo fue desarrollado con las limitaciones de
<strong>Google Colab</strong>.</p>

<p><strong>NumPy</strong> es una biblioteca de Python utilizada principalmente para operaciones
numéricas y matemáticas eficientes en matrices y arreglos multidimensionales. Con
respecto al proyecto su importancia se centra en la manipulación y el
preprocesamiento de los datos, los cálculos de las diferentes métricas y otras
funciones necesarias para el desarrollo de la red neuronal.</p>

<h4>4. Preprocesamiento de los datos</h4>
<p>Fue necesario realizar un preprocesamiento sobre cada una de las imágenes del conjunto de datos.
Este preprocesamiento se baso en el recorte de la imagen aislando el fondo negro logrando conservar solo el cerebro.</p>
<p>Para realizarlo se implementó una técnica [Rosebrock A, 2016] basada en hallar los
puntos extremos del contorno, logrando tomar el contorno completo del cerebro y
recortando la imagen en base al rectángulo que lo contiene por completo.</p>
<p>De este preprocesamiento se obtienen las imágenes que verdaderamente formaran parte del conjunto de datos utilizado para el entrenamiento y la validación de los datos. 
  De esta manera se dejan a un lado las partes de la imagen que no brindan ninguna información para el modelo.</p>

![preprocesamiento](https://github.com/melisamessa/CNN_BrainTumorDetection/assets/105131503/d9cb3d72-76c8-4a47-a32a-da09bff72dde)

<h4>5. Red Neuronal Convolucional de clasificación binaria</h4>
<p>Una red neuronal convolucional de clasificación binaria es un tipo de red neuronal diseñada específicamente para resolver problemas donde el objetivo es realizar una clasificación asignando una de dos etiquetas posibles a cada entrada. Utiliza capas convolucionales para extraer características importantes de los datos de entrada y capas de pooling para reducir la dimensionalidad de las características extraídas. Finalmente posee una capa de salida final que consta de una única neurona y produce como salida un valor perteneciente al rango [0, 1].</p>

<h4>5.1 Tratamiento de los datos</h4>
<p>Para el entrenamiento de la red fue necesario brindarle tanto datos normales como datos anómalos, para ello se conformó un dataset con 3000 imágenes preprocesadas. 
  Además al unir todas las imágenes, a los datos normales se les asignó la etiqueta 1 y a los datos anómalos la etiqueta 0.</p>
<p>Una vez con todas las asignaciones, el dataset conformado y las etiquetas asignadas, se designó el 80% de los datos para realizar el entrenamiento de la red y 
el 20% para hacer el testeo. Los valores de entrenamiento y de testeo fueron a su vez normalizados para tener valores entre 0 y 1, de esta manera el modelo de 
aprendizaje profundo se generaliza más rápido y produce mejores resultados.</p>

<h4>5.2 Definición del modelo</h4>
<p>Como el presente trabajo es de clasificación binaria, lo primero a tener en cuenta es que la capa final de la red convolucional debe ser una única capa densa de la que 
se obtiene una única salida y dicha salida representará efectivamente el valor 0 o el valor 1 gracias a la activación “sigmoid”.</p>
<p>Como el conjunto de datos son imágenes, es importante además tener varías capas convolucionales, en el presente proyecto se utilizan tres. Estás capas tendrán una 
  cantidad determinada de filtros y una activación “relu”. La función relu se encarga de que cualquier salida negativa se convierta en cero y de que las salidas positivas 
  mantengan su valor, esto permite tener en cuenta patrones no lineales. 
  Además se utilizaron las capas “flatten” y “dropout” para reducir los datos. Y se incorporó una capa densa con sesenta y cuatro neuronas que está completamente conectada 
  con la capa final antes mencionada.</p>
<p>Para la función de pérdida de una clasificación binaria se utiliza binary cross-entropy y se decide en el presente proyecto usar el optimizador RMSprop ya que genera 
  mejores resultados que el optimizador adam, este último es el que suele utilizarse pero para el conjunto de datos propuesto no fue el indicado. Como métrica se 
  decidió verificar el accuracy ya que está brinda información sobre la precisión que posee el modelo.</p>


<h4>6. Métricas</h4>
<h4>6.1 Accuracy</h4>
<p>Accuracy (exactitud) es una métrica común y muy utilizada para el aprendizaje automático y las redes neuronales. Está representa la fracción de ejemplos 
  clasificados correctamente entre el total de ejemplos. Es la proporción de predicciones correctas en relación al número total de predicciones. 
  Se expresa con valores entre 0 y 1, donde 1 significa que todas las predicciones fueron correctas y 0 que ninguna predicción fue correcta.</p>
<p>Validation accuracy (exactitud de validación) es el accuracy calculado en el conjunto de datos pertenecientes a la validación, estos son datos que el modelo no tuvo en 
  cuenta durante el entrenamiento y se utiliza para evaluar que tan bien generaliza el modelo ante datos nuevos.</p>
<p>Estas dos métricas se comparan mediante gráficos para evaluar el rendimiento del modelo a lo largo del tiempo durante el entrenamiento.</p>

![grafico_accuracy](https://github.com/melisamessa/CNN_BrainTumorDetection/assets/105131503/6f2f5314-5cd4-468f-b33c-ded5098a8f9c)

<p>En el gráfico de la figura 2 se visualizan dos líneas, la amarilla pertenece al accuracy en el conjunto de entrenamiento y la roja al accuracy en el conjunto de validación. 
  Se puede visualizar como ambas líneas aumentan y se van acercando una a la otra a medida que aumenta la cantidad de épocas de entrenamiento. Se pueden ver además algunos picos 
  pero el comportamiento que presentan las métricas es el mencionado anteriormente.</p>
<p>Gracias al gráfico se deduce que no habrá overfitting (sobreajuste) ya que las líneas tienden a subir juntas y no se da que la roja descienda mientras la amarilla asciende.</p>

<h4>6.2 Loss</h4>
<p>Loss (pérdida) es una medida fundamental para el entrenamiento de modelos de redes neuronales. Representa una medida de qué tan diferentes son las predicciones del modelo en comparación a las etiquetas reales del conjunto de datos. El objetivo del modelo debe ser minimizar está pérdida ya que esto implicaría que realiza predicciones más precisas.</p>
<p>Validation loss (pérdida de validación) es la métrica antes mencionada pero aplicada al conjunto de validación que son datos que en el entrenamiento el modelo nunca vio. 
  Está métrica se compara con la anterior mencionada para evaluar cómo se comporta el modelo durante el entrenamiento y detectar signos de sobreajuste.</p>

![grafico_loss](https://github.com/melisamessa/CNN_BrainTumorDetection/assets/105131503/d608e8cb-b4e0-4df0-be0c-291303ff7edb)

<p>En el gráfico de la figura 3 se visualizan la pérdida en el entrenamiento, línea amarilla, y la pérdida en la validación, línea roja.</p>
<p>No hay signo de sobreajuste ya que la línea de pérdida de validación no crece mientras la otra desciende. Tampoco se mantienen ambas métricas altas lo que indicaría que el modelo era demasiado simple para detectar los patrones.</p>
<p>Se puede ver como ambas líneas van disminuyendo gradualmente, pese a algunos picos que presenta la línea de validación, lo que indica que el modelo está aprendiendo bien y generalizando correctamente.</p>

<h4>7. Testeo de los datos y resultados obtenidos</h4>
<p>Una vez que se determinó que el modelo era bien entrenado y que las métricas brindaron información importante para esperar buenos resultados, se comenzó a usar el 20% de los datos guardados para el testeo.</p>
<p>Se utilizaron 600 imágenes para testear que estaban almacenadas en el arreglo de testeo y se realizó una secuencia de instrucciones para obtener las predicciones del modelo al brindarle una nueva imagen que no se utilizó ni en el proceso de entrenamiento ni en el proceso de validación.</p>
<p>Se puede determinar gracias a las etiquetas que las imágenes que obtengan una predicción con valor cercano a cero serán anómalas y las que obtengan una predicción con valor cercano a uno serán normales.</p>
<p>Se realizaron entonces múltiples pruebas para visualizar el desempeño del modelo. Primeramente se eligió una imagen normal y una anómala y se testeo individualmente, el resultado obtenido fue el siguiente:</p>

![resultados](https://github.com/melisamessa/CNN_BrainTumorDetection/assets/105131503/435e136a-4ffa-4f2e-b64c-c9d4dd09d0f4)

<p>Luego se implementó una secuencia de instrucciones para obtener la predicción e imagen de los primeros 10 datos pertenecientes al testeo:</p>

![resultados2](https://github.com/melisamessa/CNN_BrainTumorDetection/assets/105131503/404d5768-593b-4c81-88f8-733736e903e1)

<p>En las figuras 4 y 5 presentadas, se pueden observar resultados asertivos tanto para el testeo individual como para el testeo de las 10 primeras imágenes.</p>
<p>Se puede determinar entonces que el modelo implementado en el presente proyecto realiza un trabajo efectivo sobre el dataset de imágenes elegido y obtiene buenos resultados en la detección de tumores cerebrales en imágenes médicas. Logrando entonces el objetivo presentado en este proyecto, la detección de la anomalía de manera eficaz. Se debería tener en cuenta para una siguiente fase de proyecto la integración de mayor cantidad de imágenes con tumores pequeños, ya que se puede notar una limitación cuando la anomalía es diminuta, las predicciones en estos casos se encuentran entre 0.40 y 0.60, obteniendo en alguno de estos casos un resultado erróneo.</p>

<h4>8. Análisis: Autoencoder vs CNN binaria</h4>
<h4>8.1 Autoencoder</h4>
<p>Un autoencoder es una red neuronal entrenada para copiar su entrada en su salida. Se puede considerar que la red consta de dos partes: un codificador y un decodificador que produce la reconstrucción.</p>
<p>El encoder, lado izquierdo de la red, transforma la entrada original en una representación de menor dimensión. Está representación retiene solo las características relevantes de los datos, lo cual ayuda a identificar patrones significativos y a capturar la información esencial. Como salida del encoder se obtiene el llamado espacio latente, que es el resultado del entrenamiento donde la red aprende a extraer la información más relevante de los datos de entrada.</p>
<p>El decoder, lado derecho de la red, se encarga de recrear la entrada original utilizando como punto de partida la salida del decoder. Es decir, intenta invertir el proceso de codificación, buscando recrear un dato de mayor dimensión a partir de un dato de entrada de menor dimensión.</p>
<p>Los autoencoders están diseñados para no aprender a copiar perfectamente, se obliga al codificador y al decodificador a trabajar juntos para encontrar la forma más eficiente de condensar los datos de entrada en una dimensión más baja. La única forma en que funcionan es forzando la pérdida de información con el cuello de botella que se forma exactamente en el medio de la red neuronal, este cuello de botella es el resultado de la compresión de la entrada. El autoencoder está diseñado para minimizar el error de reconstrucción que surge de la diferencia entre la entrada original y la reconstrucción de la misma.</p>
<p>Estas redes son útiles para la reducción de dimensión o compresión, para limpiar una imagen con ruido y para detectar anomalías.</p>

![diagrama_autoencoder](https://github.com/melisamessa/CNN_BrainTumorDetection/assets/105131503/0de47627-44a3-4e83-8226-e9735b0b3d43)

<h4>8.2 Detección de anomalías</h4>
<p>Utilizando autoencoders podemos identificar anomalías en los datos, cuando estos no están etiquetados. Está estrategia no utiliza el autoencoder en sí mismo sino que utiliza el error de reconstrucción producido al revertir la codificación de la entrada.</p>
<p>El error de reconstrucción para la detección de anomalías se basa en métodos de reducción de dimensionalidad que permiten proyectar las características relevantes de la entrada en un espacio de menor dimensión que el original, conservando la mayor información posible.</p>
<p>El autoencoder posee una función que mapea la posición que ocupa cada observación en el espacio original con el nuevo espacio generado, y solo las observaciones que fueron bien proyectadas son capaces de volver a la posición que ocupaban originalmente con una precisión elevada.</p>
<p>En este caso los datos anómalos serán mal proyectados y por tanto su error de reconstrucción será elevado cumpliendo con el objetivo de la detección de la anomalía.</p>

<h4>8.3 Elección de arquitectura</h4>
<p>Para la implementación del Autoencoder se utilizó el mismo conjunto de datos que se utilizó para la red convolucional de clasificación binaria. Está arquitectura fue la primera prueba del proyecto, sin embargo brindaba resultados erróneos.</p>
<p>Al momento de utilizar la comparación de los errores de reconstrucción de las imágenes anómalas con las imágenes normales, las imágenes normales obtenían un valor mayor. Esto conllevo a que la arquitectura no lograra su objetivo.</p>
<p>Pese a agregar más datos para el entrenamiento, al preprocesamiento de los datos, a cambiar la arquitectura del modelo, el autoencoder no logró su cometido.</p>
<p>Uno de los motivos por el que la arquitectura del autoencoder no fue efectiva, se debe al conjunto de datos utilizados, teniendo en cuenta que la real función de un autoencoder es comprimir y descomprimir datos de entrada, y que no fue diseñado específicamente para la detección de anomalías.</p>
<p>Por ello se decidió implementar una segunda arquitectura, la red neuronal convolucional de clasificación binaria con la que se obtuvieron los resultados deseados como se demuestra en la sección 6 del presente informe.</p>

<h4>9. Extensiones</h4>
<h4>9.1 Red neuronal convolucional binaria para la clasificación de tumores benignos y malignos</h4>
<p>Realizar una clasificación de imágenes de tumores benignos y tumores malignos verificando similitudes y diferencias entre ellos. Utilizando una gran cantidad de datos que en la actualidad no se encuentran en Internet público, para lograr un proyecto que hoy en día aún no existe. La determinación de si un tumor es maligno o benigno se realiza mediante biopsias y no mediante imágenes a simple vista, por lo tanto está idea es un gran desafío.</p>

<h4>9.2 Mejora de precisión</h4>
<p>Investigar enfoques avanzados de aprendizaje profundo, arquitecturas más complejas o incluso datos adicionales para mejorar la precisión de la detección. Realizar la clasificación planteada con la misma implementación pero teniendo en cuenta obtener imágenes de cerebros sanos e imágenes de cerebros con tumores pequeños. Muy pequeños, ya que esta es la limitación que posee el presente proyecto. Estás imágenes tampoco se encuentran en cantidad en Internet público, por ende sería necesario obtener los datos de algún sitio privado.</p>

<h4>9.3 Detección y clasificación de nuevas anomalías</h4>
<p>Se puede extender el proyecto para lograr una clasificación de los diferentes tipos de tumores cerebrales como pueden ser meningiomas, tumores pituitarios, glioblastoma, entre otros. Además de poder detectar otras anomalías como hemorragias, demencia, alzheimer, epilepsia, entre otras, que también se presentan en imágenes de resonancia magnética.</p>

<h4>Bibliografía</h4>

*   Rosebrock A. (2016). Finding extreme points in contours with OpenCV.
*   Beggle L, Pfeiffer M, Bischl B. (2019). Robust Anomaly Detection in Images using Adversarial Autoencoders.
*   TensorFlow. Intro to Autoencoders.
*   Rodrigo J. A. (2021). Detección de Anomalías con Autoencoders y Python.
*   Goodfellow I, Bengio Y, Courville A. (2015). Deep Learning. Chapter 14.
*   Temas de ciencia y tecnología. (2012). Análisis de imágenes de mamografía para la detección de cáncer de mama. Vol. 15, nro. 47, pp 39-45.
*   Jácobo-Zavaleta S, Zavaleta J. (2023). A Deep Learning Approach for Epilepsy Seizure Identification using Electroencephalogram Signals: A Preliminary Study.
