import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
import json
from datetime import datetime
from typing import Optional
import gzip
import re

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

@app.get("/genre/{genero}")
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
        
        return {f'El género {genero} se encuentra en el puesto número': posicion}
    else:
        return {'El género brindado no se encuentra en la base de datos, por favor revíselo'}


@app.get("/userforgenre/{genero}")
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
        return {'El género brindado no se encuentra en la base de datos, por favor revíselo'}

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

#http://127.0.0.1:8000