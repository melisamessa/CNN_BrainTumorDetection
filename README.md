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

<h4>5. Red Neuronal Convolucional de clasificación binaria</h4>
<p>Una red neuronal convolucional de clasificación binaria es un tipo de red neuronal
diseñada específicamente para resolver problemas donde el objetivo es realizar una
clasificación asignando una de dos etiquetas posibles a cada entrada. Se basa en la
arquitectura de una red neuronal convolucional pero que posee una capa de salida
final que consta de una única neurona y produce como salida un valor perteneciente
al rango [0, 1].</p>

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


<h4>6.2 Loss</h4>
