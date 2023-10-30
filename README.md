# PI_ML_OPS_Recomendacion_Juegos_Steam

# <h1 align=center> **Proyecto Individual MLOps** </h1>
# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>
<p align="center">
<img src="Images/img_1.png"  height=300>
</p>

# **Introducción : Steam Game Recomendation**

En este proyecto, tendremos que realizar una serie de transformaciones para la obtención de información útil y un modelo capaz de recomendar juegos similares a los que le gustaron al usuario. 
Para esto habrá que realizar todos los procesos necesarios presentes en el ciclo de vida de un dato desde la carga, el tratamiento hasta la selección de los componentes o variables que se utilizaran en nuestro Modelo de Machine Learning para su posterior despliegue y uso.

Cabe aclarar que el proyecto fue desarrollado en el lapso de una semana y, por tanto, puede tener mejoras pero lo que se busca es un Minimun Viable Product(MVP) que se entragará a producción.

Adoptaremos el rol de una persona que es contratada como Data Scientist en una start-up que provee servicios de agregación de plataformas streaming. Nuestro objetivo principal será crear un modelo de Machine Learning capaz de proporcionar recomendaciones de juegos a nuestros clientes.

La madurez de los datos que tenemos no es óptima, hay datos sin transformar ,anidados, muchos datos faltantes y no tenemos un proceso automatizado para dar con los mismos por lo que tendremos que realizar una serie de transformaciones para obtener información.


# **Objetivos**

Nuestros objetivos principales son la generación de una API (Application Programming Interfaces) con una serie de funciones para su uso, el Deployment o despliegue de la misma para su uso y la realización de un modelo de recomendación de juegos en la plataforma Steam  con Machine Learning.

-------------------------------------------------------------------------------------

**MENÚ:**

:heavy_check_mark: Carpeta ETL_EDA_Feature_Eng: 
1.  Archivo ProjectoSteam_ETL.ipynb :
        En este archivo se realizan la carga de los archivos originales, casi todas las tranformaciones de datos , se exporta una version casi sin tranformaciones nombre_raw
        de cada archivo para usar en el EDA y se continua con las tranformaciones, luego se los
        exporta limpios.

2. Archivo EDA.ipynb :
        En este archivo se realizan la visualizaciones de los 3 archivos y del 4 resultante del Featuring_Engineering.

3. Archivo Featuring_Engineering.ipynb :
        En este archivo se reemplzan las reseñas por el sentimiento de cada reseña usando Machine Learning(NLP).

4. Sistema_Recomendacion.ipynb :
        En este archivo se explica el sistema de recomendacion como funciona y se muestra ejemplos de uso.    

5. Funciones_API.ipynb :
        En contiene las funciones de la API con pruebas de distintos inputs , explicaciones y mas comentarios sobre el codigo     

6. prueba.ipynb :
        Contiene pruebas de las funciones de la API con explicaciones del modelo de Machine Learning usado para la recomendacion

:heavy_check_mark: Carpeta Datasets: Esta carpeta contiene los datasets necesarios para correr
la API , que habian sido exportados ya el proceso de ETL 
             

:heavy_check_mark: Archivo 'README.md' - en este readme encontrará especificaciones de todo el proyecto.

-------------------------------------------------------------------------------------

### Se listaron una serie de tareas para llevar a cabo el proceso de ETL, las describiremos brevemente a continuación:

 ### **Pasos A Seguir**

 Comenzaremos cargando los datos , desanidandolos.
 Seguiremos con un Analisis Exploratorio de Datos (EDA) para ver la composicion de los datos su distribucion y relaciones.

Posteriormente se realiza un limpieza de datos nulos y atributos innecesarios para nuestro MVP
Y para finalizar las tranformaciones de datos haremos un Featuring Engineering usando Machine Learning para hacer una Sentiment Analysis , catalogando las reviews en Negativas,Neutras y 
Positivas.

Para finalizar construiremos una API que contiene consultas sobre los datos y un sistema de 
recomendación user-item de juegos de Steam

En este link encontraras las tranformaciones realizadas y el EDA  el cual fue realizado en la nube (Google Colab): [ETL_EDA](https://github.com/PKleinDataSets/PI_ML_OPS_Recomendacion_Juegos_Steam/tree/main/ETL_EDA_Feature_Eng)

# **Desarrollo de nuestra API**

En esta etapa del proyecto, se propone montar y desplegar una API que responda a las peticiones del usuario disponibilizando la data de la empresa mediante el uso
de el Framework FastAPI.

Se han definido 6 funciones para los endpoints que serán consumidos en la API, cada una de ellas con un decorador `@app.get('/')`.

### A continuación, se detallan las funciones y las consultas que pueden realizarse a través de la API:


1. + def **developer( *`desarrollador` : str* )**:
    `Cantidad` de items y `porcentaje` de contenido Free por año según empresa desarrolladora. 
Ejemplo de retorno:

| Año  | Cantidad de Items | Contenido Free  |
|------|-------------------|------------------|
| 2023 | 50                | 27%              |
| 2022 | 45                | 25%              |
| xxxx | xx                | xx%              |


2. + def **userdata( *`User_id` : str* )**:
    Debe devolver `cantidad` de dinero gastado por el usuario, el `porcentaje` de recomendación en base a reviews.recommend y `cantidad de items`.

Ejemplo de retorno: {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}

3. + def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
			     "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
	
4. + def **best_developer_year( *`año` : int* )**:
   Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

5. + def **developer_reviews_analysis( *`desarrolladora` : str* )**:
    Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo. 

Ejemplo de retorno: {'Valve' : [Negative = 182, Positive = 278]}


6. + Sistema de recomendación user-item:
    + def **recomendacion_usuario( *`id de usuario`* )**:
        Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.


En este archivo encontrarás la implementación de FastAPI y el desarrollo de las funciones: [main](https://github.com/PKleinDataSets/PI_ML_OPS_Steam2/blob/main/main.py)

# **Link a la API y instrucciones:**

+ [Deployment](https://app2-dho4.onrender.com/docs#/): El proyecto ha sido desplegado en un entorno en línea utilizando **Render**

:warning: **Sintaxis a tener en cuenta al escribir una consulta:** :warning:

:red_circle: Los generos deben estar escritos con la primera letra en mayúscula y el resto
en minuscula  y sin comillas , ejemplo : Action, Simulation, Strategy.

:red_circle: Los años deben ser escritos sin comillas:
    ejemplo : 2013, 2014, 2015 , etc.

:red_circle: el user_id debe ser escrito de forma exacta respetando mayúsculas, minúsculas
y el resto de caracteres. Ejemplo : barbas1, Z1_M4N, bobseagull.

:link: En este [link](https://app2-dho4.onrender.com/docs) podras ingresar y consultar las funciones. Cliqueando en GET de cada función, luego la opción 'Try it out', colocar los el dato correspondiente, y cliquear botón 'Execute'.


-------------------------------------------------------------------------------------

:wrench: **EN ESTE PROTECTO SE UTILIZARON LAS SIGUIENTES HERRAMIENTAS**

- Python.
- Visual studio code.
- Google Colaboratory
- Librería pandas.
- Librería numpy.
- Librería seaborn.
- Librería matplotlib.
- Librería datetime.
- Librería scikit-learn.
- Librería nltk.
- Librería fastparquet.
- FastApi.
- Uvicorn (servidor web).
- Render (plataforma de deployment de API)

-------------------------------------------------------------------------------------


# **Archivos de interés , LINK A VIDEO DE YOUTUBE y fuentes:**

+ [Dataset original](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj): Aquí encontrarás una carpeta con los archivos usados

+ [Diccionario de datos](https://docs.google.com/spreadsheets/d/1-t9HLzLHIGXvliq56UE_gMaWBVTPfrlTf2D9uAtLGrk/edit?usp=drive_web&ouid=108742311118546721132): Diccionario con algunas descripciones de las columnas disponibles en el dataset.

+ [Video explicativo](https://youtu.be/RUVuZAiKzg0) Un video Youtube explicando algunos conceptos sobre el proyecto, y mostrando el deployment de la API en Render.

+ [API](https://app2-dho4.onrender.com/docs#/)  API del proyecto.

:fire: [Gmail](pablodatasets@gmail.com) : pablodatasets@gmail.com Mail de contacto

:fire: [LinkedIn](https://www.linkedin.com/in/pablo-andres-klein-a669551b1/) : Mi perfil de linkedin

Gracias por la visita :smile: :bangbang:

Autor: Pablo Andrés Klein

