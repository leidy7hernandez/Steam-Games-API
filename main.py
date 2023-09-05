import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
import json
from datetime import datetime
from typing import Optional
import gzip
import re
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

# Para leer el archivo json.gzip
with gzip.open('data_items.json.gz', 'rb') as archivo_json_comprimido:
    users_items = pd.read_json(archivo_json_comprimido, lines=True, orient='records')
with gzip.open('data_reviews.json.gz', 'rb') as archivo_json_comprimido:
    user_reviews = pd.read_json(archivo_json_comprimido, lines=True, orient='records')
with gzip.open('output_steam_games.json.gz', 'rb') as archivo_json_comprimido:
    steam_games = pd.read_json(archivo_json_comprimido, lines=True, orient='records')

@app.get("/")
def welcome():
    return {'descripción':'El objetivo de esta API es principalmete mostrar los resultados para las siguientes funciones:',
            'def userdata( User_id : str )':'Devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.',
            'def countreviews( YYYY-MM-DD , YYYY-MM-DD : str )':'Cantidad de usuarios que realizaron reviews entre las fechas dadas y, el porcentaje de recomendación de los mismos en base a reviews.recommend.',
            'def genre( género : str )':'Devuelve el puesto en el que se encuentra un género sobre el ranking de los mismos analizado bajo la columna PlayTimeForever.',
            'def userforgenre( género : str )':'Retorna el top 5 de usuarios con más horas de juego en el género dado, con su URL (del user) y user_id.',
            'def developer( desarrollador : str )':'Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.',
            'def sentiment_analysis( año : int )':'Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.',
            'def recomendacion_juego(product_id : int)':'Ingresando el id de producto, muestra una lista con 5 juegos recomendados similares al ingresado',
            'Instrucciones':'Para llamar cada función por favor utiliza "/nombre_de_la_funcion/input"'}

@app.get('/userdata/{user_id}')
async def userdata(user_id: str):
    user_id = user_id.lower()
    
    # Verificar si el usuario existe en users_items
    if user_id not in users_items['user_id'].values:
        return {'Message': 'El usuario que brinda no se encuentra en la base de datos.'}
    
    # Filtrar las revisiones del usuario
    user_reviews_filtered = user_reviews[user_reviews['user_id'] == user_id]

    # Calcular el porcentaje de recomendación
    total_reviews = user_reviews_filtered['reviews'].explode().dropna()
    total = len(total_reviews)
    recomendados = total_reviews.apply(lambda review: review['recommend'] == 'True').sum()
    porcentaje_recom = (recomendados / total) * 100

    # Filtrar datos de users_items
    user_items_data = users_items[users_items['user_id'] == user_id]

    # Extraer información
    spend = round(user_items_data['total_amount'].values[0], 2)
    items = user_items_data['items_count'].values[0]

    return {'Porcentaje de recomendación de juegos': float(porcentaje_recom),
            'Total de dinero gastado por el usuario': float(spend),
            'Cantidad Total de items': float(items)}

def is_valid_date(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

@app.get("/countreviews/{first_date}/{last_date}")
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
        
        return {f'Cantidad de usuarios que realizaron reviews entre las fechas {first_date} y {last_date} es de': usuarios_count,
            f'Porcentaje de juegos recomendados entre las fechas {first_date} y {last_date} es de': round(porcentaje_recom, 2)}
    else:
        return {'El rango de fechas dado está por fuera del de la base de datos. Se encuentra información desde 2010-10-16 hasta 2015-12-31'}

def genres_list():
    # Descomponer las listas de géneros en filas separadas
    exploded_genres = steam_games['genres'].explode()
    # Obtener los géneros únicos y eliminar los valores nulos
    unique_genres = exploded_genres.dropna().unique()
    return unique_genres

def playtime_per_genre():
    g_list = genres_list()
    # Crear un diccionario para realizar un seguimiento de las sumas por género
    sumas_por_genero = {genero: 0 for genero in g_list}

    # Recorrer el DataFrame y actualizar las sumas por género
    for i in range(len(users_items)):
        item_list = users_items['items'][i]

        for item_data in item_list:
            item_name = item_data['item_name']
            playtime = item_data.get('playtime_forever', 0)

            if isinstance(item_name, list):
                for genero in item_name:
                    if genero in sumas_por_genero:
                        sumas_por_genero[genero] += playtime

    # Crear un diccionario final con los resultados
    resultados_por_genero = {genero: sumas_por_genero[genero] for genero in g_list}
    diccionario_ordenado = dict(sorted(resultados_por_genero.items(), key=lambda item: item[1], reverse=True))
    return diccionario_ordenado

@app.get("/genre/{genero}")
def genre( genero : str ):
    genero = genero.lower()
    ranking = playtime_per_genre()

    if genero in list(a.keys()):
        # Obtener la posición del género en el ranking
        posicion = list(ranking.keys()).index(genero)

        return {f"La posición de {genero} en el ranking es": posicion+1}
    else:
        return{'Message':f'{genero} no se encuentra en el ranking, por favor revíselo'}


@app.get("/userforgenre/{genero}")
def userforgenre(genero):
    genero = genero.lower()
    g_list = genres_list()
    
    if genero in g_list:
        # Crear un diccionario para almacenar las sumas por usuario
        sumas_por_usuario = {}

        # Filtrar y sumar playtime solo para usuarios con 'simulation' en item_name
        for i in range(len(users_items)):
            user_id = users_items['user_id'][i]
            for item_data in users_items['items'][i]:
                item_name = item_data.get('item_name', [])
                playtime = item_data.get('playtime_forever', 0)
                if isinstance(item_name, list) and genero in item_name:
                    if user_id in sumas_por_usuario:
                        sumas_por_usuario[user_id] += playtime
                    else:
                        sumas_por_usuario[user_id] = playtime

        # Ordenar el diccionario de sumas por usuario en orden descendente
        diccionario_ordenado = dict(sorted(sumas_por_usuario.items(), key=lambda item: item[1], reverse=True))

        # Obtener el top 5 de usuarios con mayor playtime en 'simulation'
        top_5 = dict(list(diccionario_ordenado.items())[:5])

        # Crear un diccionario para almacenar las URLs de los usuarios
        urls_por_usuario = {}

        # Mapear las URLs de los usuarios en el top 5
        for user_id in top_5:
            index = users_items[users_items['user_id'] == user_id].index[0]
            urls_por_usuario[user_id] = users_items['user_url'][index]

        return (urls_por_usuario)
    else:
        return {'Message':'El género brindado no se encuentra en la base de datos, por favor revíselo'}
    
@app.get("/developer/{desarrollador}")
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

        if cantidad != 0:
            return {'Años': años,
                    'Porcentaje de contenido free por año': porcentaje_free_por_año}
    else: 
        return{'El desarrollador brindado no se encuentra en la base de datos, por favor revíselo'}

@app.get("/sentiment_analysis/{anio}")
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
        return {'El año dado está por fuera del de la base de datos. Se encuentra información desde 2010-10-16 hasta 2015-12-31'}

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

@app.get("/recomendacion_juego/{product_id}")
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

#http://127.0.0.1:8000