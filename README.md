# %% [markdown]
# ## API PARA RECOMENDACIÓN DE VIDEOJUEGOS EN STEAM 

# %% [markdown]
# 

# %% [markdown]
# ## ETL (Extraction, Transformation, Load)

# %% [markdown]
# * Librerias necesarias

# %%
import pandas as pd
import numpy as np
import re
import json
import gzip

# %% [markdown]
# Al realizar la cargar normal de los archivos se presentaban errores porque estaban en formato gzip, es por eso que antes de generar codigo éstos se extrajeron en la misma carpeta para evitar futuras complicaciones y optimizar el código.

# %% [markdown]
# Además, dos archivos archivos tienen sólo comilla simple (') en lugar de comillas dobles ("), es decir, los archivos estan corruptos y adicionalmente anidados. Esto también genera un error al crear el dataframe así que se prefiere extraer y limpiar los archivos uno por uno para evitar problemas futuros.

# %% [markdown]
# -- steam_games

# %% [markdown]
# Con el archivo steam_games no había ningún problema así que este es el primero que se carga:

# %%
# Cargar el archivo JSON en un DataFrame
steam_games = pd.read_json('output_steam_games.json', lines=True)

# %% [markdown]
# -- user_reviews

# %% [markdown]
# Ahora, para cargar (o más bien crear) el archivo user_reviews es necesario hacer algunas modificaciones, y es por eso que se prefiere guardar línea por línea para poder aplicar dichos cambios. ACLARACIÓN: Se prefiere realizar el proceso de limpieza de forma manual debido a que los recursos que se tienen para manipular los datos son muy escasos.

# %%
jlines = []
# Abrir el archivo JSON en modo lectura
with open('australian_user_reviews.json', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Guardar las líneas del archivo JSON
for line in lines:
    jlines.append(line)

# %% [markdown]
# Durante el proceso de ETL se pudo evidenciar tres casos puntuales donde en "reviews" se encontró respuestas que afectaban el procedimiento de limpieza de datos de forma negativa (se comentará más adelante el por qué), y es por esto que se decide hacer una modificación manual de "{" por "(" y de "}" por ")".

# %%
for i in range(len(jlines)):
    jlines[i] = jlines[i].replace('{LINK REMOVED}',"(LINK REMOVED)")
    jlines[i] = jlines[i].replace('{誰是殺手}',"(誰是殺手)")
    jlines[i] = jlines[i].replace("}{@r|)c0r3",")(@r|)c0r3")

# %% [markdown]
# Para tener un archivo json limpio se decide hacer un diccionario con las mismas columnas originales y luego con este crear un nuevo archivo, dado que es más efectivo y seguro que reemplazar el archivo de user_reviews. Para esto hay que extraer la información de las columnas en el archivo json original; las principales columnas son user_id, user_url y reviews.

# %%
# Extraer la información de las tres columnas principales del archivo json original
user_id = []
user_url = []
reviews = []

for line in jlines:
    match = re.search(r"'user_id':\s+'([^']+)'[^}]*'user_url':\s+'([^']+)'[^}]*'reviews':\s+(\[.*\])", line)
    if match:
        user_id.append(match.group(1))
        user_url.append(match.group(2))
        reviews.append(match.group(3))

# %% [markdown]
# Seguidamente notamos que la columna reviews a su vez tiene columnas con información útil, así que extraemos los datos de la columna reviews.

# %% [markdown]
#     Como se puede ver en el código, el requerimiento para extraer la información de reviews es que los caracteres estén entre "{" y "}", es por esto que comentarios como "{LINK REMOVED}" afectaba negativamente a nuestro proceso de organización. Por lo tanto, al ser sólo tres casos puntuales se decide cambiar los "{ }" por "()" respectivamente rn fichas reseñas.

# %%
extracted_data_reviews = []

for review in reviews:
    # Encuentra los datos entre '[' y ']' en cada revisión -> Para extraer todos los caracteres de reviews
    matches = re.findall(r'\{(.*?)\}', review)
    for match in matches:
        extracted_data_reviews.append('{'+match+'}')


# %% [markdown]
# Una vez con los datos individuales de cada columna limpios y listos procedemos a extraer la información de todas las columnas de reviews.

# %% [markdown]
#     Como algunos comentarios de review presentan comillas es necesario hacer una extraccion aparte para esta columna.

# %%
funny = []
posted = []
last_edited = []
item_id = []
helpful = []
recommend = []

for line in extracted_data_reviews:
    match_funny = re.search(r"'funny':\s+'([^']+)'[^}]", line)
    match_posted = re.search(r"'posted':\s+'([^']+)'", line)
    match_last_edited = re.search(r"'last_edited':\s+'([^']+)'", line)
    match_item_id = re.search(r"'item_id':\s+'([^']+)'[^}]", line)
    match_helpful = re.search(r"'helpful':\s+'([^']+)'[^}]", line)
    match_recommend = re.search(r"'recommend':\s+([a-zA-Z]+)", line)
    if match_funny:
        funny.append(match_funny.group(1))
    if not match_funny:
        funny.append("")
    if match_posted:
        posted.append(match_posted.group(1))
    if match_last_edited:
        last_edited.append(match_last_edited.group(1))
    if not match_last_edited:
        last_edited.append("")
    if match_item_id:
        item_id.append(match_item_id.group(1))
    if match_helpful:
        helpful.append(match_helpful.group(1))
    if match_recommend:
        recommend.append(match_recommend.group(1))

# %%
review = []
for i in range(len(extracted_data_reviews)):
    data = extracted_data_reviews[i]

    # Encontrar la posición del inicio de la reseña
    review_start = data.find("'review': ") + len("'review': ")

    # Encontrar la posición del final de la reseña
    review_end = data.find("}", review_start)

    # Extraer el texto de la reseña
    review_text = data[review_start:review_end]

    # Reemplazar comillas dobles escapadas con comillas dobles normales
    review_text = review_text.replace('\'\'', '\'').replace('"',"'")
    review.append(review_text)

for i in range(len(review)):
    if review[i].startswith("'") and review[i].endswith("'"):
        review[i] = review[i][1:-1]

# %% [markdown]
# Finalmente tenemos todos los datos limpios, y si no se ha notado, una vez que éstos se extraen se les agrega las comillas dobles (") de manera automática, entonces se ha cumplido el objetivo principal de la transformación del archivo user_reviews. Sin embargo, también se nota que dentro de reviews hay varios comentarios, por ende "funny", que es la primera columna de reviews se repite varias veces; para asegurarnos que todo esté bien se hace una lista que contenga la cantidad de veces que aparezca "funny" en cada fila de reviews.

# %%
len(reviews)

# %%
len(review)

# %%
counter = []
for i in range(len(reviews)):
    count_funny = reviews[i].count("'funny':")
    counter.append(count_funny)

# %% [markdown]
# Ahora comprobamos que la suma de nuestro contador y que la cantidad de datos de funny sea la misma. Como todo está en orden entonces estamos listos para formar nuestro diccionario, pero nótese que se deben formar dos diccionarios: Uno para reviews y otro para data_reviews en general, dado que reviews en sí tiene columnas que organizar.

# %%
len(funny) - sum(counter)

# %% [markdown]
# Cada fila de nuestro archivo final debe tener aproximadamente la siguiente estructura:

# %% [markdown]
#     data_reviews = {
#         "user_id" : user_id[i],
#         "user_url" : user_url[i],
#         "reviews" : reviews_modified[i]
#     }

# %% [markdown]
# Primero organizamos el diccionario para reviews:

# %% [markdown]
# Se crea la función form_review que se encarga de crear el diccionario linea por linea para cada review. Finalmente se recopilan los reviews en un diccionario final.

# %%
def form_review(count):
    t = sum(counter[:counter[count]])
    reviews_line = []
    for i in range(sum(counter[:count]), sum(counter[:count])  + counter[count]):
        new_review = {
            "funny" : funny[i],
            "posted" : posted[i],
            "last_edited" : last_edited[i],
            "item_id" : item_id[i],
            "helpful" : helpful[i],
            "recommend" : recommend[i],
            "review" : review[i]
        }
        reviews_line.append(new_review)
    return reviews_line

# %%
reviews_modified = []
for i in range(len(counter)):
    reviews_modified.append(form_review(i))

# %% [markdown]
# Ahora se crea el diccionario final para nuestro archivo json ...

# %%
data_reviews = []
for i in range(len(user_id)):
    data = {
        "user_id" : user_id[i],
        "user_url" : user_url[i],
        "reviews" : reviews_modified[i]
    }  
    data_reviews.append(data)

# %%
# Guardar los datos en un archivo JSON
with open('data_reviews.json', 'w') as json_file:
    for review_data in data_reviews:
        json.dump(review_data, json_file)
        json_file.write('\n')

# %% [markdown]
# ... y finalmente cargamos esto en nuestro DataFrame user_reviews

# %%
# Cargar el archivo JSON en un DataFrame
user_reviews = pd.read_json('data_reviews.json', lines=True)

# %% [markdown]
# -- users_items

# %% [markdown]
# Ahora, necesitamos hacer el mismo proceso de limpieza pero para crear el archivo users_items. Se sigue un proceso muy similar que con user_reviews, por no decir que es prácticamente el mismo. 

# %% [markdown]
#     No se presentaron cambios significativos en el código, así que no se comentará este extracto por extracto.

# %%
jlines = []
# Abrir el archivo JSON en modo lectura
with open('australian_users_items.json', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Guardar las líneas del archivo JSON
for line in lines:
    jlines.append(line)

# %%
user_id = []
items_count = []
steam_id = []
user_url = []

# Patrón de búsqueda usando comillas simples en lugar de dobles
pattern = r"'user_id':\s+'([^']+)'[^}]*'items_count':\s+(\d+)[^}]*'steam_id':\s+'([^']+)'[^}]*'user_url':\s+'([^']+)'"

for i in range(len(jlines)):
    # Contenido de la línea con comillas simples
    line = jlines[i]

    match = re.search(pattern, line)
    if match:
        user_id.append(match.group(1))
        items_count.append(int(match.group(2)))
        steam_id.append(match.group(3))
        user_url.append(match.group(4))

# %%
items = []

for i in range(len(jlines)):
    # Contenido de la línea con comillas simples
    line = jlines[i]

    # Encontrar la posición del inicio de la lista de items
    items_start = line.find("\'items\': ") + len("\'items\': ")

    # Extraer la lista de items (incluyendo los corchetes)
    items_info = line[items_start:]

    # Eliminar el corchete final si está presente
    if items_info.endswith("}"):
        items_info = items_info[:-1]
    
    items.append(items_info)

# %%
extracted_data_items = []

for item in items:
    # Encuentra los datos entre '[' y ']' en cada revisión -> Para extraer todos los caracteres de reviews
    matches = re.findall(r'\{(.*?)\}', item)
    for match in matches:
        extracted_data_items.append('{'+match+'}')

# %%
item_id = []
item_name = []
playtime_forever = []
playtime_2weeks = []

# Línea de ejemplo
for i in range(len(extracted_data_items)):
    line = extracted_data_items[i]
    match_item_id = re.search(r"'item_id':\s+'([^']+)'[^}]", line)
    match_item_name = re.search(r"'item_name':\s+\"([^\"]+)\"", line)
    match_playtime_forever = re.search(r"'playtime_forever':\s+(\d+)[^}]", line)
    match_playtime_2weeks = re.search(r"'playtime_2weeks':\s+(\d+)", line)

    if match_item_id:
        item_id.append(match_item_id.group(1))
    if match_item_name:
        item_name.append(match_item_name.group(1))
    if not match_item_name:
        match_item_name = re.search(r"'item_name':\s+'([^']+)'", line)
        if match_item_name:
            item_name.append(match_item_name.group(1))
    if match_playtime_forever:
        playtime_forever.append(match_playtime_forever.group(1))
    if match_playtime_2weeks:
        playtime_2weeks.append(match_playtime_2weeks.group(1))


# %%
counter = []
for i in range(len(items)):
    count_item_id = items[i].count("'item_id':")
    counter.append(count_item_id)

# %%
def form_item(count):
    t = sum(counter[:counter[count]])
    item_line = []
    for i in range(sum(counter[:count]), sum(counter[:count])  + counter[count]):
        new_item = {
            "item_id" : item_id[i],
            "item_name" : item_name[i],
            "playtime_forever" : playtime_forever[i],
            "playtime_2weeks" : playtime_2weeks[i]
        }
        item_line.append(new_item)
    return item_line

# %%
items_modified = []
for i in range(len(counter)):
    items_modified.append(form_item(i))

# %%
data_items = []
for i in range(len(user_id)):
    data = {
        "user_id" : user_id[i],
        "items_count" : items_count[i],
        "steam_id" : steam_id[i],
        "user_url" : user_url[i],
        "items" : items_modified[i]
    }  
    data_items.append(data)

# %%
# Guardar los datos en un archivo JSON
with open('data_items.json', 'w') as json_file:
    for items_data in data_items:
        json.dump(items_data, json_file)
        json_file.write('\n')

# %%
# Cargar el archivo JSON en un DataFrame
users_items = pd.read_json('data_items.json', lines=True)

# %% [markdown]
# ## NLP (Natural Language Processing)

# %% [markdown]
# Ahora, ya se tienen los datos limpios pero al ser tantos y tener recursos tan limitados lo mejor sería encontrar una forma de optimizar dichos datos. Se ve potencial en reemplazar las reseñas de user_reviews['reviews']..['reviews'] por un análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna debe reemplazar la de user_reviews.review para facilitar el trabajo de los modelos de machine learning y el análisis de datos. 

# %% [markdown]
# * Librerias necesarias 

# %%
from langdetect import detect
import re
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# %% [markdown]
# Para hacer el NLP el primer obstáculo que presentamos es que algunos de las reseñas se encuentran en otros idiomas diferentes del inglés (que es el idioma en que tenemos los datos), así que a modo de exploración nos conviene saber la cantidad de frases que no se encuentran en inglés.

# %%
language = []
index = []
phrases = []
pattern = re.compile(r'^[a-zA-Z]+$')

for i, text in enumerate(review):
    if bool(pattern.match(text)):
        detected_language = detect(text)
        if detected_language != 'en':
            language.append(detected_language) 
            index.append(i) # Es una excelente idea saber en qué indice dento de reviews se encuentra ubicada la reseña
            phrases.append(review[i])


# %% [markdown]
# Dado que tenemos un número considerable de reseñas en un idioma diferente al inglés, estaría bien no perder esos datos sino más bien traducir dichas reseñas al inglés y luego reemplzar dichos comentarios.

# %% [markdown]
#     Si bien muchas frases identificadas en language como que pertenecen a otro idioma en realidad no es así, esto no afecta en nada porque finalmente se van a traducir al inglés.

# %%
translated = []
for i in range (len(phrases)):
    if len(phrases[i]) < 5000:    # Hay un comentario no útil que tienen más de 5000 caracteres
        traductor = GoogleTranslator(source=language[i], target='en')
        translated.append(traductor.translate(phrases[i]))
    else:
        translated.append(phrases[i])

# %%
# Reemplazar las traducciones de las reseñas
for i in range(len(translated)):
    review[index[i]] = translated[i]

# %% [markdown]
# Ahora que se tiene todas las reseñas en inglés se vuelve a formar el archivo json pero con la información modificada.

# %%
counter = []
for i in range(len(reviews)):
    count_funny = reviews[i].count("'funny':")
    counter.append(count_funny)

# %%
reviews_modified = []
for i in range(len(counter)):
    reviews_modified.append(form_review(i))

# %%
# Extraer la información de las tres columnas principales del archivo json original
user_id = []
user_url = []
reviews = []

for line in jlines:
    match = re.search(r"'user_id':\s+'([^']+)'[^}]*'user_url':\s+'([^']+)'[^}]*'reviews':\s+(\[.*\])", line)
    if match:
        user_id.append(match.group(1))
        user_url.append(match.group(2))
        reviews.append(match.group(3))

# %%
data_reviews = []
for i in range(len(user_id)):
    data = {
        "user_id" : user_id[i],
        "user_url" : user_url[i],
        "reviews" : reviews_modified[i]
    }  
    data_reviews.append(data)

# %% [markdown]
# Ya que se tienen todas las reseñas en el idioma adecuado, ahora se hace el modelaje de NLP. Se crean las variables y se entrena el modelo, como lo que se quiere es clasificar las reseñas como positivas, negativas o neutrales, un clasificador de Naive Bayes Gaussiano sería el preciso para ésta tarea.

# %%
df_reviews = pd.DataFrame(review) # Se crea el DataFrame de reviews para que se puedan procesar todos los datos
cv = CountVectorizer(max_features = 1500)# solo toma 1500 palabras
X = cv.fit_transform(df_reviews[0]).toarray() 
y = df_reviews.iloc[:, -1].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, random_state = 0)

# %%
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# %% [markdown]
# A continuación se procede con realizar un análisis de sentimiento en cada reseña. Primero se inicializan las herramientas sia, ps y all_stopwords; luego se crea la función analyze_review y finalmente se realiza la clasificación.

# %%
# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer() 

# Inicializar el stemmer
ps = PorterStemmer()

# Obtener la lista de stopwords en inglés
all_stopwords = set(stopwords.words('english'))

def analyze_review(new_review):
    # Preprocesamiento del texto
    new_review = re.sub('[^a-zA-Z]', ' ', new_review.lower())
    new_review = new_review.replace(" t ", "ot ")
    new_review = ' '.join([ps.stem(word) for word in new_review.split() if word not in all_stopwords])

    # Convertir el texto preprocesado en matriz
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()

    # Calcular el puntaje de sentimiento
    sentiment_score = sia.polarity_scores(new_review)['compound']

    # Determinar el sentimiento basado en el puntaje
    if sentiment_score > 0:
        sentiment = 2  # Positivo
    elif sentiment_score < 0:
        sentiment = 0  # Negativo
    else:
        sentiment = 1  # Neutral"""
    return sentiment

# %%
classification = []
for i in range(len(review)):
    classification.append(analyze_review(review[i]))

# %% [markdown]
# Ahora que tenemos la clasificación de cada reseña, se reemplaza dicha reseña por su calificación. Seguidamente se guardan los datos en el archivo json.

# %%
# Crear un diccionario que mapea las revisiones a sus clasificaciones correspondientes
mapeo_revisiones = {review[i]: classification[i] for i in range(len(review))}

# Recorrer el DataFrame user_reviews_copy una sola vez
for i in range(len(user_reviews)):
    for j in range(len(user_reviews['reviews'][i])):
        review_actual = user_reviews['reviews'][i][j]['review']
        # Verificar si la revisión actual está en el diccionario de mapeo
        if review_actual in mapeo_revisiones:
            # Reemplazar la revisión con su clasificación correspondiente
            user_reviews['reviews'][i][j]['review'] = mapeo_revisiones[review_actual]


# %%
# Guardar los datos en un archivo JSON
with open('data_reviews.json', 'w') as json_file:
    for review_data in data_reviews:
        json.dump(review_data, json_file)
        json_file.write('\n')

# %%
# Guardar el DataFrame en un archivo JSON
user_reviews.to_json('data_reviews.json', orient='records', lines=True)

# %% [markdown]
# ## Modificaciones previas

# %% [markdown]
# Antes de hacer las funciones para la Api, es útil modificar los archivos y eliminar todas las columnas que no se utilizan para garantizar que el espacio y el rendimiento.

# %% [markdown]
#     steam_games

# %%
del steam_games['publisher']
del steam_games['title']
del steam_games['url']
del steam_games['tags']
del steam_games['specs']
del steam_games['early_access']
del steam_games['reviews_url']

# %%
# Define una función personalizada para reemplazar 'Free' o 'free' con 0
def replace_free_with_zero(price):
    if price is not None and isinstance(price, str) and ("Free" in price or "free" in price):
        return 0
    return price

# Define una función personalizada para reemplazar valores específicos con 0
def replace_specific_values(price):
    if price is not None and isinstance(price, str):
        if "Starting at $449.00" in price:
            return 449
        elif "Starting at $499.00" in price:
            return 499
        elif price in ['Play WARMACHINE: Tactics Demo', 'Install Theme', 'Play the Demo', 'Play Now', 'Install Now', 'Third-party']:
            return 0
    return price

# Aplica ambas funciones a la columna 'price'
steam_games['price'] = steam_games['price'].apply(replace_free_with_zero)
steam_games['price'] = steam_games['price'].apply(replace_specific_values)

# %%
# Identifica las filas donde todas las columnas son None
rows_to_remove = steam_games[steam_games.isnull().all(axis=1)].index

# Luego, utiliza el método 'drop' para eliminar estas filas del DataFrame
steam_games = steam_games.drop(rows_to_remove)

# Resetear los index
steam_games.reset_index(drop=True, inplace=True)

# %%
# Es mejor estandarizar todos los datos en minúsculas
steam_games['app_name'] = steam_games['app_name'].str.lower()
steam_games['developer'] = steam_games['developer'].str.lower()
for i in range(len(steam_games)):
    genres_list = steam_games['genres'][i]
    if genres_list is not None:
        steam_games['genres'][i] = [genre.lower() for genre in genres_list]

# %%
# También es mejor estandarizar todos los id en int

# Convierte la columna 'id' a números, tratando los valores no numéricos como NaN
steam_games['id'] = pd.to_numeric(steam_games['id'], errors='coerce')

# Llena los valores NaN con un valor por defecto (por ejemplo, -1)
steam_games['id'] = steam_games['id'].fillna(-1).astype(int)


# %%
steam_games.to_json('output_steam_games.json', orient='records', lines=True)

# %% [markdown]
#     user_reviews

# %%
#Función para modificar las fechas 
def update_date(original_date):
    try:
        parsed_date = datetime.strptime(original_date, "%B %d %Y")
        formatted_date = parsed_date.strftime("%Y-%m-%d")
        return formatted_date
    except ValueError:
        pass

# %%
# Se aplica la función update_date para reemplazar las fechas en user_reviews
for i in range(len(user_reviews)):
    for j in range(len(user_reviews['reviews'][i])):
        if user_reviews['reviews'][i][j]['posted'] == None:
            pass
        else:
            user_reviews['reviews'][i][j]['posted'] = update_date(user_reviews['reviews'][i][j]['posted'].replace(',','').replace('.','').replace('Posted ',''))

# %%
# Eliminar datos innecesarios dentro de reviews (funny, last_edited,helpful)
for i in range(len(user_reviews)):
    for j in range(len(user_reviews['reviews'][i])):
        del user_reviews['reviews'][i][j]['funny']

for i in range(len(user_reviews)):
    for j in range(len(user_reviews['reviews'][i])):
        del user_reviews['reviews'][i][j]['last_edited']

for i in range(len(user_reviews)):
    for j in range(len(user_reviews['reviews'][i])):
        del user_reviews['reviews'][i][j]['helpful']

# %%
user_reviews.to_json('data_reviews.json', orient='records', lines=True)

# %% [markdown]
#     users_items

# %%
# Define una función para extraer valores numéricos de una cadena
def extract_price(text):
    match = re.search(r'\d+\.\d+', str(text))
    return float(match.group()) if match else 0.0

# Crea un diccionario para mapear los nombres de los juegos a sus precios
game_prices = dict(zip(steam_games['app_name'], steam_games['price'].apply(extract_price)))

# Agrega una nueva columna 'total_amount' a users_items e inicialízala con 0
users_items['total_amount'] = 0.0

# Calcula el monto total gastado por cada usuario y actualiza la columna 'total_amount'
for i in range(len(users_items)):
    user_id = users_items['user_id'][i]
    total_spent = 0
    
    for item in users_items['items'][i]:
        item_name = item['item_name']
        
        # Busca el precio del juego en el diccionario game_prices
        price = game_prices.get(item_name, 0)
        total_spent += price
    
    # Actualiza el valor en la columna 'total_amount' para el usuario actual
    users_items.at[i, 'total_amount'] = total_spent

# %%
for i in range(len(users_items)):
    for j in range(len(users_items['items'][i])):
        del users_items['items'][i][j]['item_id']

# %%
for i in range(len(users_items)):
    for j in range(len(users_items['items'][i])):
        del users_items['items'][i][j]['playtime_2weeks']

# %%
# El tipo de dato int pesa menos que str, para ahorrar espacio, se va a convertir playtime_forever a int
for i in range(len(users_items)):
    for j in range(len(users_items['items'][i])):
        users_items['items'][i][j]['playtime_forever'] = int(users_items['items'][i][j]['playtime_forever'])

# %% [markdown]
#     Para seguir ahorrando espacio en la memoria, es útil agrupar los juegos por su género y sumar en ellos el tiempo total de playtime_forever

# %%
# Definir una función para convertir una cadena a minúsculas y eliminar apóstrofes
def procesar_texto(texto):
    if isinstance(texto, str):
        # Convertir a minúsculas y eliminar apóstrofes
        texto = texto.lower().replace("'", "").replace('"','').replace("-"," ")
    return texto

# Aplicar la función a todo el DataFrame
steam_games = steam_games.applymap(procesar_texto)

# %%
# Para que todos los datos sean en minúsculas en la columna genres
nuevas_listas_de_generos = []
for genres_list in steam_games['genres']:
    if genres_list is not None:
        nueva_lista = [procesar_texto(genre) for genre in genres_list]
        nuevas_listas_de_generos.append(nueva_lista)
    else:
        nuevas_listas_de_generos.append(None)

steam_games['genres'] = nuevas_listas_de_generos

# %%
# Aplicar la función a todo el DataFrame
users_items = users_items.applymap(procesar_texto)

# %%
# Función para convertir a minúsculas y procesar el texto
def procesar_texto(texto):
    if isinstance(texto, str):
        # Convertir a minúsculas y eliminar apóstrofes, comillas y guiones
        texto = texto.lower().replace("'", "").replace('"', '').replace("-", " ").replace("  ", " ")
    return texto

# Función para aplicar procesar_texto a un diccionario
def procesar_item(item):
    return {key: procesar_texto(value) if isinstance(value, str) else value for key, value in item.items()}

# Aplicar la función procesar_item y convertir a minúsculas a todos los elementos de la columna 'items'
users_items['items'] = users_items['items'].apply(lambda lista: [procesar_item(item) for item in lista])

# %%
#Para reemplazar el nombre del juego por el género
# Crear un diccionario de mapeo de nombres de elementos a géneros
element_to_genre = {game['app_name']: game['genres'] for _, game in steam_games.iterrows()}

# Iterar a través de users_items y actualizar los nombres con géneros
for i in range(len(users_items)):
    for j in range(len(users_items['items'][i])):
        item_name = users_items['items'][i][j]['item_name']
        if item_name in element_to_genre:
            users_items['items'][i][j]['item_name'] = element_to_genre[item_name]

# %%
# Hacer una lista de géneros
genres_list = []

for genres in steam_games['genres']:
    if genres is not None:
        genres_list.extend(genres)

genres_list = list(set(genres_list))

# %%
def get_genre_names(item_name):
    if item_name is not None:
        return item_name
    return []

# %%
# Agrupar como diccionario los datos restantes
for i in range(len(users_items)):
    if len(users_items['items'][i]) > 0:
        lista_de_ceros = np.zeros(len(genres_list))
        df = pd.DataFrame(users_items['items'][i])
        df['item_name'] = df['item_name'].apply(get_genre_names)

        for k in range(len(df)):
            for j in range(len(genres_list)):
                if genres_list[j] in df['item_name'][k]:
                    lista_de_ceros[j] += df['playtime_forever'][k]

        # Crear una nueva lista con los elementos que no son cero
        nueva_lista_de_ceros = [x for x in lista_de_ceros if x != 0]
        # Crear una nueva lista de géneros correspondientes
        nuevos_genres_list = [genre for i, genre in enumerate(genres_list) if lista_de_ceros[i] != 0]

        # Asignar las nuevas listas a las originales
        lista_de_ceros = nueva_lista_de_ceros
        genres_list = nuevos_genres_list

        for j in range(len(lista_de_ceros)):
            lista_de_ceros[j] = int(lista_de_ceros[j])

        diccionario = dict(zip(genres_list, lista_de_ceros))

        users_items['items'][i] = diccionario
    else:
        pass

# %%
# Guardar el DataFrame en un archivo JSON
users_items.to_json('data_items.json', orient='records', lines=True)

# %% [markdown]
#     Los archivos aún siguen pesando mucho, es por eso que se van a comprimir

# %%
# Para comprimir el archivo
with open('data_items.json', 'rb') as archivo_json:
    with gzip.open('data_items.json' + '.gz', 'wb') as archivo_json_comprimido:
        archivo_json_comprimido.writelines(archivo_json)

# %%
# Para leer el archivo json.gzip
with gzip.open('data_items.json.gz', 'rb') as archivo_json_comprimido:
    users_items = pd.read_json(archivo_json_comprimido, lines=True, orient='records')

# %%
with open('data_reviews.json', 'rb') as archivo_json:
    with gzip.open('data_reviews.json' + '.gz', 'wb') as archivo_json_comprimido:
        archivo_json_comprimido.writelines(archivo_json)

# %%
# Para leer el archivo json.gzip
with gzip.open('data_reviews.json.gz', 'rb') as archivo_json_comprimido:
    user_reviews = pd.read_json(archivo_json_comprimido, lines=True, orient='records')

# %%
with open('output_steam_games.json', 'rb') as archivo_json:
    with gzip.open('output_steam_games.json' + '.gz', 'wb') as archivo_json_comprimido:
        archivo_json_comprimido.writelines(archivo_json)

# %%
# Para leer el archivo json.gzip
with gzip.open('output_steam_games.json.gz', 'rb') as archivo_json_comprimido:
    steam_games = pd.read_json(archivo_json_comprimido, lines=True, orient='records')

# %% [markdown]
# ## Funciones alimentadoras de la API

# %% [markdown]
# Ahora se desea hacer una API para facilitar algunas consultas en específico. 

# %% [markdown]
# * Librerias necesarias

# %%
from datetime import datetime

# %% [markdown]
# La primera función es def userdata( User_id : str ): La cual devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.

# %%
def userdata( User_id : str ): 
    User_id = User_id.lower()
    users = list(set(users_items['user_id']))
    if User_id in users:

        # Calcular el porcentaje de recomendación
        total = 0
        recomendados = 0
        for _, row in user_reviews[user_reviews['user_id'] == User_id].iterrows():
            for review in row['reviews']:
                if review['recommend'] == 'True':
                    recomendados += 1
                total += 1
        porcentaje_recom = (recomendados / total) * 100 

        for i in range(len(users_items)):
            if users_items['user_id'][i] == User_id:
                spend = round(users_items['total_amount'][i],2)
                items = users_items['items_count'][i]

        print(f'Porcentaje de recomendación de juegos: {porcentaje_recom:.2f}%\n'
            f'Total de dinero gastado por el usuario:{spend}\n'
            f'Cantidad Total de items: {items}')
    else:
        print('El usuario que brinda no se encuentra en la base de datos.')

# %%
userdata('js41637')

# %% [markdown]
# Ahora se hace la función def countreviews(first_date, last_date : str ): La cual retorna la cantidad de usuarios que realizaron reviews entre dos fechas dadas y, el porcentaje de recomendación de los mismos en base a reviews.recommend. Primero se hace un ajuste en el formato de las fechas, luego se ejecuta la función is_valid_date para identificar si las fechas están completas, y finalmente se crea la funcion countreviews.

# %%
def is_valid_date(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# %% [markdown]
#     También se debe saber cuál es la primera y última fecha en reviews para que la función retorne valores dentro de ese mismo rango de fechas

# %%
dates = [review['posted'] for reviews_list in user_reviews['reviews'] for review in reviews_list]
fechas_validas = [fecha for fecha in dates if fecha is not None]
fechas_datetime = [datetime.strptime(fecha, '%Y-%m-%d') for fecha in fechas_validas]
print(f'Primera fecha:{min(fechas_datetime)}')
print(f'última fecha:{max(fechas_datetime)}')

# %%
def countreviews(first_date, last_date : str ): 
    if datetime.strptime(first_date, '%Y-%m-%d') > datetime.strptime('2010-10-16', '%Y-%m-%d') and datetime.strptime(last_date, '%Y-%m-%d') < datetime.strptime('2015-12-31', '%Y-%m-%d') :
        usuarios = set()
        total = 0
        recomendados = 0
        
        for i in range(len(user_reviews)):
            for review in user_reviews['reviews'][i]:
                posted_date = review['posted']
                if posted_date and is_valid_date(posted_date):
                    if first_date <= posted_date <= last_date:
                        usuarios.add(user_reviews['user_id'][i])
                        if review['recommend'] == "True":
                            total += 1
                            recomendados += 1
                        else:
                            total += 1
        
        usuarios_count = len(usuarios)
        porcentaje_recom = (recomendados / total) * 100
        
        print(f'Cantidad de usuarios que realizaron reviews entre las fechas {first_date} y {last_date} es de: {usuarios_count}\n'
            f'Porcentaje de juegos recomendados entre las fechas {first_date} y {last_date} es de: {round(porcentaje_recom, 2)}%')
    else:
        print('El rango de fechas dado está por fuera del de la base de datos. Se encuentra información desde 2010-10-16 hasta 2015-12-31')


# %%
countreviews('2012-01-01','2013-01-01')

# %% [markdown]
# Ahora se crea la función def genre( género : str ): Que devuelve el puesto en el que se encuentra un género sobre el ranking de los mismos analizado bajo la columna PlayTimeForever.

# %%
def genres_list():
    # Descomponer las listas de géneros en filas separadas
    exploded_genres = steam_games['genres'].explode()
    # Obtener los géneros únicos y eliminar los valores nulos
    unique_genres = exploded_genres.dropna().unique()
    return unique_genres

# %%
def genre(genero: str):
    genero = genero.lower()
    # Accede a los tiempos de juego por género desde users_items['items'][0]
    playtime_data = users_items['items'][0]
    
    if genero in playtime_data:
        # Obtiene el tiempo de juego para el género dado
        tiempo_jugado = playtime_data[genero]
        
        # Obtiene la lista de géneros únicos utilizando la función genres_list
        unique_genres = genres_list()
        
        # Ordena los géneros por tiempo de juego en orden descendente
        ranking = sorted(unique_genres, key=lambda x: playtime_data.get(x, 0), reverse=True)
        
        # Encuentra la posición del género en el ranking
        posicion = ranking.index(genero) + 1
        
        print(f'El género {genero} se encuentra en el puesto número {posicion} sobre el ranking de los géneros con {tiempo_jugado} minutos de tiempo jugado')
    else:
        print('El género brindado no se encuentra en la base de datos, por favor revíselo')


# %%
genre('action')

# %% [markdown]
# También se hace la función def userforgenre( género : str ): Que retorna el top 5 de usuarios con más horas de juego en el género dado, con su URL (del user) y user_id.

# %%
def userforgenre(genero):
    genero = genero.lower()
    g_list = genres_list()
    
    if genero in g_list:
        # Inicializa un diccionario para almacenar el tiempo de juego por usuario
        playtime_by_user = {}

        # Recorre el DataFrame users_items
        for index, row in users_items.iterrows():
            user_id = row['user_id']
            user_items = row['items']

            # Inicializa el tiempo de juego para este usuario en 0
            total_playtime = 0

            # Verifica si el género está presente en los items del usuario
            if genero in user_items:
                total_playtime = user_items[genero]

            # Agrega el tiempo de juego al diccionario
            playtime_by_user[user_id] = total_playtime

        # Ordena el diccionario en función del tiempo de juego en orden descendente
        sorted_users = sorted(playtime_by_user.items(), key=lambda x: x[1], reverse=True)

        # Toma los primeros 5 usuarios del ranking
        top_5_users = sorted_users[:5]

        # Crea un diccionario con la información de los usuarios en el formato requerido
        top_5_user_info = {}
        for user_id, playtime in top_5_users:
            user_url = users_items[users_items['user_id'] == user_id]['user_url'].values[0]
            top_5_user_info[user_id] = user_url

        return top_5_user_info
    else:
        print('El género brindado no se encuentra en la base de datos, por favor revíselo')


# %%
userforgenre('ACTION')

# %% [markdown]
# Otra de las funciones es def developer( desarrollador : str ): Que debe retornar la cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.

# %%
def developer(desarrollador: str):
    desarrollador = desarrollador.lower()
    """Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. Ejemplo de salida:"""
    developers = list(set(steam_games['developer']))
    if desarrollador in developers:
        años = []
        cantidad = 0
        free = 0
        
        for i in range(len(steam_games)):
            if steam_games['developer'][i] == desarrollador:
                cantidad += 1
                if steam_games['price'][i] in ['Free Demo', 'Free Mod', 'Free to Use', 'Free To Play', 'Free Movie', 'Play for Free!', 'Free to Try', 'Free', 'Free to Play', 'Free HITMAN™ Holiday Pack']:
                    free += 1
                if steam_games['release_date'][i] is not None and is_valid_date(steam_games['release_date'][i]):
                    año = steam_games['release_date'][i][:4]
                    años.append(año)
        años = list(set(años))  # Eliminar duplicados
        porcentaje_free_por_año = (free/cantidad)*100

        print(f'Cantidad total de juegos: {cantidad}')
        if cantidad != 0:
            return {'Años': años,
                    'Porcentaje de contenido free por año': porcentaje_free_por_año}
    else: 
        return{'El desarrollador brindado no se encuentra en la base de datos, por favor revíselo'}

# %%
developer('ExtinctionArTS')

# %% [markdown]
# Finalmente, se hace la funcion def sentiment_analysis( año : int ): Que según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.

# %%
def sentiment_analysis( año : int ): 
    if año > 2009 and año < 2016:
        positivo = 0
        negativo = 0
        neutral = 0
        for i in range(len(user_reviews)):
            for j in range(len(user_reviews['reviews'][i])):
                if user_reviews['reviews'][i][j]['posted'] is None:
                    pass
                if user_reviews['reviews'][i][j]['posted'] is not None:
                    if int(user_reviews['reviews'][i][j]['posted'][:4]) == año:
                        if user_reviews['reviews'][i][j]['review'] == 2:
                            positivo += 1
                        elif user_reviews['reviews'][i][j]['review'] == 1:
                            neutral += 1
                        else:
                            negativo += 1

        resultado = {'Positivo':positivo, 'Neutral':neutral,'Negativo':negativo}
        return resultado
    else:
        print('El año dado está por fuera del de la base de datos. Se encuentra información desde 2010-10-16 hasta 2015-12-31')

# %%
sentiment_analysis(2014)

# %% [markdown]
# ## Construcción de la API

# %% [markdown]
# * Primero se crea el entorno virtual con el comando python -m venv proyecto-env

# %% [markdown]
# * Se activa el entorno virtual: proyecto-env\Scripts\activate.bat

# %% [markdown]
# * Se instala fastapi: pip install fastapi

# %% [markdown]
# * Instalar uvicorn: pip install "uvicorn[standard]"

# %% [markdown]
# * Hacer el freeze de los requirements: pip freeze > requirements.txt --> Si luego se necesita instalar otra librería más, se vuelve a ejecutar este comando.

# %% [markdown]
# * Se crea el archivo main.py y se importa FastAPI: 

# %% [markdown]
#     from fastapi import FastAPI
# 
#     app = FastAPI()

# %% [markdown]
# * Se forman todas las funciones necesarias

# %% [markdown]
# * Levantar el servidor: python -m uvicorn main:app --reload

# %% [markdown]
# ## Exploratory Data Analysis-EDA (Análisis exploratorio de los datos)

# %% [markdown]
# Librerias Necesarias

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# Para hacer el EDA vamos a tomar la posición de Steam, de allí saldrán preguntas interesantes como:
# - ¿Cómo es la distribución del precio de los juegos?
# - ¿Cuál es el género de juegos más gustado por la gente y qúe estudios pertenecen a ese género?
# - ¿Cuál es la relación entre el review y la cantidad de tiempo jugado?
# - Top 10 de los usuarios que más reseñas dejan - Dado que son posibles streamers o influencers, hay que tenerlos en cuenta 
# - ¿Cuál es el mes en que más se lanzan juegos? - Visualización de ciclos

# %% [markdown]
#     ¿Cómo es la distribución del precio de los juegos?

# %% [markdown]
# Esta sección del EDA es importante para identificar si hay juegos muy costosos y poco accesibles para la comunidad de jugadores

# %%
# Calcular estadísticas
media = steam_games['price'].mean()
mediana = steam_games['price'].median()
desviacion_estandar = steam_games['price'].std()
minimo = steam_games['price'].min()
maximo = steam_games['price'].max()
moda = steam_games['price'].mode()

print("Media:", media)
print("Mediana:", mediana)
print("Desviación Estándar:", desviacion_estandar)
print("Mínimo:", minimo)
print("Máximo:", maximo)
print("Moda:", moda)

# %%
# Eliminar filas con valores None en la columna 'developer'
steam_games_cleaned = steam_games.dropna(subset=['developer'])

# Extraer las columnas 'X' e 'Y' para el gráfico de dispersión
x = steam_games_cleaned['developer']
y = steam_games_cleaned['price']

# Crear el gráfico de dispersión
plt.scatter(x, y)

# Personalizar el gráfico (opcional)
plt.title('Gráfico de Dispersión')
plt.ylabel('Precio')

plt.xticks([])

# Mostrar el gráfico
plt.show()


# %% [markdown]
# Se puede evidenciar que la mayoría de juegos está dentro del rango de $0 a $100, muy pocos juegos salen de dicho rango, y sólo un juego cuesta casi los mil dólares ($995.00)

# %% [markdown]
#     ¿Cuál es el género de juegos más gustado por la gente y qúe estudios pertenecen a ese género?

# %% [markdown]
# Si utilizamos la función genre,se puede ver que el género más jugado por los usuarios es Action. 

# %%
# Crear un DataFrame a partir del diccionario de desarrolladores por género
developer_counts_df = pd.DataFrame.from_dict(developer_count_by_genre, orient='index', columns=['Cantidad de Desarrolladores'])

# Ordenar el DataFrame por la cantidad de desarrolladores en orden descendente
developer_counts_df = developer_counts_df.sort_values(by='Cantidad de Desarrolladores', ascending=False)

# Crear un gráfico de barras
plt.figure(figsize=(12, 6))  # Tamaño del gráfico
plt.bar(developer_counts_df.index, developer_counts_df['Cantidad de Desarrolladores'], color='skyblue')
plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mayor legibilidad
plt.xlabel('Género')
plt.ylabel('Cantidad de Desarrolladores')
plt.title('Cantidad de Desarrolladores por Género')
plt.tight_layout()  # Ajustar el diseño para evitar cortar etiquetas

# Mostrar el gráfico
plt.show()


# %% [markdown]
# Si bien action es un juego muy popular entre los jugadores, los desarrolladores prefieren producir juegos indies. 

# %% [markdown]
#     ¿Cuál es la relación entre el la recomendación y el precio del juego?

# %% [markdown]
# Es interesante ver si de alguna forma con juegos más económicos dicho juego es más o menos recomendado entre los usuarios

# %%
recommend = []
for i in range(len(user_reviews)):
    counter = 0
    for j in range(len(user_reviews['reviews'][i])):
        if user_reviews['reviews'][i][j]['recommend'] == "True":
            counter += 1
    recommend.append(counter)

# %%
# Crear un DataFrame para organizar los datos 
df = pd.DataFrame()

df['user_id'] = user_reviews['user_id']
df['recommend_list'] = recommend

# %%
# Crear un diccionario con user_id como clave y total_amount como valor
user_amount_dict = dict(zip(users_items['user_id'], users_items['total_amount']))

# Inicializar una lista para almacenar los valores de total_amount correspondientes en df
total = []

# Iterar a través de df y obtener los valores de total_amount del diccionario
for user_id in df['user_id']:
    total_amount = user_amount_dict.get(user_id, 0)  # Usar 0 como valor predeterminado si no se encuentra el user_id
    total.append(total_amount)

df['total_amount'] = total

# %%
# Calcular la correlación entre las columnas "recommend_list" y "total_amount"
correlation_matrix = df[['total_amount','recommend_list']].corr()
plt.figure(figsize=(6, 4))  # Para ajustar el tamaño del gráfico

# Se utiliza sns.heatmap para crear el mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Mapa de Calor de Correlación')
plt.show()


# %% [markdown]
# Como la correlación entre el precio del juego y el porcentaje de recomendación es altamente positiva, significa que a medida que el precio del juego aumenta, el porcentaje de recomendación también tiende a aumentar. En otras palabras, los juegos más caros tienden a recibir una mayor proporción de recomendaciones positivas.
# 
# - Explicación posible: Los juegos más caros pueden ofrecer características adicionales, gráficos de alta calidad, contenido adicional o una experiencia de juego más completa. Como resultado, los jugadores que han invertido más dinero en un juego pueden estar más satisfechos con su compra y, por lo tanto, es más probable que recomienden el juego a otros.

# %% [markdown]
#     Top 10 de los usuarios que más reseñas dejan

# %% [markdown]
# Es importante identificar si dentro de la comunidad de usuarios hay posibles streamers, ellos afectan en la opinión de los demás o potenciales usuarios.

# %% [markdown]
# Primero abría que analizar si la cantidad de reviews es significativa para algunos usuarios.

# %%
cant_reviews = []
for i in range(len(user_reviews)):
    cant_reviews.append(len(user_reviews['reviews'][i]))

# %%
set(cant_reviews)

# %% [markdown]
# Como se puede ver, no hay usuarios con una cantidad de reseñas significativamente alta (ningún usuario pasá la frontera de 10 reseñas), es por eso que se puede deducir que no hay posibles streamers o influencers dentro de este grupo de usuarios

# %% [markdown]
#     ¿Cuál es el mes en que más se lanzan juegos? 

# %% [markdown]
# Es interesante ver cuál es el mes en el que lanzan más juegos, esto podría ser útil para llevar un conteo de los posibles gastos en marketing para esos meses

# %%
steam_games['release_date'] = pd.to_datetime(steam_games['release_date'])
steam_games['mes_lanzamiento'] = steam_games['release_date'].dt.month
meses_mas_lanzamientos = steam_games.groupby('mes_lanzamiento').size()

# %%
import matplotlib.pyplot as plt

# Crear un gráfico de barras para mostrar la cantidad de juegos por mes
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
plt.bar(meses, meses_mas_lanzamientos)

# Personalizar el gráfico
plt.title('Cantidad de Juegos Lanzados por Mes en Steam')
plt.xlabel('Mes')
plt.ylabel('Cantidad de Juegos Lanzados')

# Mostrar el gráfico
plt.show()


# %% [markdown]
# Como se puede ver, Octubre es el mes en que se lanzan más videojuegos. Ahora es interesante analizar si hay ciclos específicamente a lo largo de los años 

# %% [markdown]
# Para identificar posibles ciclos el lanzamiento mensual de juegos en Steam, puedes se puede suavizar la línea de tendencia utilizando un promedio móvil simple (SMA) en la gráfica.

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame llamado steam_games con las columnas 'release_date' y 'price'
# Asegúrate de convertir 'release_date' al tipo de dato DateTime si aún no está en ese formato
steam_games['release_date'] = pd.to_datetime(steam_games['release_date'])

# Filtra los juegos lanzados entre 2009 y 2019
start_date = pd.to_datetime('2014-01-01')
end_date = pd.to_datetime('2018-12-31')
filtered_games = steam_games[(steam_games['release_date'] >= start_date) & (steam_games['release_date'] <= end_date)]

# Ordena el DataFrame filtrado por fecha de lanzamiento
filtered_games = filtered_games.sort_values(by='release_date')

# Define la ventana del SMA (por ejemplo, 30 días para un mes)
sma_window = 30

# Calcula el SMA del precio utilizando la función rolling
filtered_games['SMA'] = filtered_games['price'].rolling(window=sma_window).mean()

# Crea el gráfico de precios y SMA
plt.figure(figsize=(12, 6))
plt.plot(filtered_games['release_date'], filtered_games['price'], label='Precio', color='blue', alpha=0.7)
plt.plot(filtered_games['release_date'], filtered_games['SMA'], label=f'SMA-{sma_window} días', color='red')

# Configura el gráfico
plt.title('Promedio Móvil Simple (SMA) del Precio de Juegos en Steam (2009-2019)')
plt.xlabel('Fecha de Lanzamiento')


# %% [markdown]
# Como se evidencia con el promedio móvil simple si hay repuntes de lanzamientos antes del útimo trimestre del año. Esto puede ser explicado con que octubre es un mes estratégico para lanzar juegos antes de la temporada de compras navideñas. Las compañías buscan aprovechar el período previo a las vacaciones, cuando las personas compran regalos, incluyendo videojuegos.

# %% [markdown]
# ## Modelo de aprendizaje automático: Sistema de recomendación

# %% [markdown]
# * Librerias necesarias:

# %%
import numpy as np
from sklearn.neighbors import NearestNeighbors

# %% [markdown]
# Ahora que toda la data es consumible por la API, está lista para consumir, y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un sistema de recomendación. Para esto se van a utilizar el algoritmo KNN, porque éste proporciona recomendaciones basadas en la similitud entre elementos o usuarios, por lo tanto es el ideal para nuestro propósito.

# %% [markdown]
# El sistema de recomendación se basa en que ingresando el id de producto se recibe una lista con 5 juegos recomendados similares al ingresado.

# %%
# Primero hay que crear una matriz de características para los juegos utilizando los géneros
def crear_matriz_caracteristicas(steam_games):
    genres = steam_games['genres']
    num_games = len(genres)
    unique_genres = list(set(genre for game_genres in genres if game_genres is not None for genre in game_genres))

    # Crear una matriz de características con valores binarios para cada género
    feature_matrix = np.zeros((num_games, len(unique_genres)))

    for i, game_genres in enumerate(genres):
        if game_genres is not None:
            for j, genre in enumerate(unique_genres):
                if genre in game_genres:
                    feature_matrix[i, j] = 1

    return feature_matrix, unique_genres

# %%
def recomendacion_juego(product_id : int):
    if product_id in list(set(steam_games['id'])):
        # Crear una matriz de características para los juegos
        genres = steam_games['genres']
        num_games = len(genres)
        unique_genres = list(set(genre for game_genres in genres if game_genres is not None for genre in game_genres))

        # Crear una matriz de características con valores binarios para cada género
        feature_matrix = np.zeros((num_games, len(unique_genres)))

        for i, game_genres in enumerate(genres):
            if game_genres is not None:
                for j, genre in enumerate(unique_genres):
                    if genre in game_genres:
                        feature_matrix[i, j] = 1

        # Crear un modelo KNeighbors
        neigh = NearestNeighbors(n_neighbors=6)  # 6 para incluir el juego de consulta
        neigh.fit(feature_matrix)

        # Encontrar el índice del juego en función del product_id
        game_index = np.where(steam_games['id'] == product_id)[0][0]

        # Encontrar los juegos más similares
        _, indices = neigh.kneighbors([feature_matrix[game_index]])

        # Obtener los nombres de los juegos recomendados
        recommended_games = [steam_games['app_name'][i] for i in indices[0][1:]]
        
        result_dict = {}
        # Crear un diccionario con el resultado en el formato deseado
        for i, juego in enumerate(recommended_games, 1):    
            result_dict[i] = juego

        return result_dict
    
    else:
        return {'Message':f'El product id {product_id} no está en la base de datos, por favor revíselo'}

# %%
recomendacion_juego(610660)

# %% [markdown]
# Ya que se tiene la última función con el sistema de recomendación, se agrega a la Api. Con esto se da por concluido nuestro trabajo para recomendar juegos en steam! :D


