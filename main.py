import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
import json
from datetime import datetime
from typing import Optional
import gzip

app = FastAPI()

users = []
# Abre el archivo comprimido en modo binario y descomprime los datos
with gzip.open('data_items.json.gz', 'rb') as archivo_json_comprimido:
    for linea in archivo_json_comprimido:
        datos_json = linea.decode('utf-8')
        objeto_json = json.loads(datos_json)
        # Añade el objeto JSON completo a la lista
        users.append(objeto_json)
# Crea un DataFrame a partir de la lista de objetos JSON
users_items = pd.DataFrame(users)
steam_games = pd.read_json('output_steam_games.json', lines=True)
user_reviews = pd.read_json('data_reviews.json', lines=True)

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

@app.get('/userdata/{User_id}')
def userdata( User_id : str ): 
    users = list(set(users_items['user_id']))
    if User_id in users:
        # Filtrar los juegos comprados por el usuario
        user_games = set()  # Evita tener datos duplicados
        for _, row in users_items[users_items['user_id'] == User_id].iterrows():
            user_games.update(item['item_name'] for item in row['items'])
        
        # Calcular el dinero gastado por el usuario
        amount_spend = 0
        for _, game_row in steam_games[steam_games['app_name'].isin(user_games)].iterrows():
            game_price = game_row['price']
            if game_price not in ['Free to Play', 'Free', None, 'Third-party','Free To Play','Free Movie','Install Now']:
                amount_spend += float(game_price)

        # Calcular la cantidad de juegos que tiene el usuario
        cant_items = len(user_games)

        # Calcular el porcentaje de recomendación
        total = 0
        recomendados = 0
        for _, row in user_reviews[user_reviews['user_id'] == User_id].iterrows():
            for review in row['reviews']:
                if review['recommend'] == 'True':
                    recomendados += 1
                total += 1
        porcentaje_recom = (recomendados / total) * 100 

        return {'La cantidad de dinero gastado por el usuario es': round(amount_spend, 2),
                'Porcentaje de recomendación de juegos': porcentaje_recom,
                'Cantidad de juegos que tiene el usuario': cant_items}
    else:
        return {'El usuario que brinda no se encuentra en la base de datos.'}

def update_date(original_date):
    try:
        parsed_date = datetime.strptime(original_date, "%B %d %Y")
        formatted_date = parsed_date.strftime("%Y-%m-%d")
        return formatted_date
    except ValueError:
        pass

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
        
        return {f'Cantidad de usuarios que realizaron reviews entre las fechas {first_date} y {last_date} es de' : usuarios_count,
            f'Porcentaje de juegos recomendados entre las fechas {first_date} y {last_date} es de': round(porcentaje_recom, 2)}
    else:
        return{'El rango de fechas dado está por fuera del de la base de datos. Se encuentra información desde 2010-10-16 hasta 2015-12-31'}


def playtime(game):
    filtered_items = [item['playtime_forever'] for items in users_items['items'] for item in items if item['item_name'] == game]
    total_playtime = sum(map(int, filtered_items))
    return total_playtime

def game_genres(game):
    matching_games = steam_games[steam_games['app_name'] == game]
    genres = [genre for genres_list in matching_games['genres'].dropna() for genre in genres_list]
    return genres

def genres_list():
    # Descomponer las listas de géneros en filas separadas
    exploded_genres = steam_games['genres'].explode()
    # Obtener los géneros únicos y eliminar los valores nulos
    unique_genres = exploded_genres.dropna().unique()
    return unique_genres

@app.get("/genre/{genero}")
def genre( genero : str ):
    g_list = genres_list()
    if genero in g_list:
        games = set(steam_games['app_name'])  # Utilizar un conjunto en lugar de una lista para buscar de manera más eficiente
        total_playtime = {}

        for i in range(len(users_items['items'])):
            for item in users_items['items'][i]:
                if 'playtime_forever' in item:
                    game_name = item['item_name']
                    playtime_forever = int(item['playtime_forever'])
                    
                    if game_name in games:
                        if game_name not in total_playtime:
                            total_playtime[game_name] = playtime_forever
                        else:
                            total_playtime[game_name] += playtime_forever

        games_by_genre = {}

        for i in range(len(steam_games)):
            app_name = steam_games['app_name'][i]
            genres = steam_games['genres'][i]
            
            if genres is not None:
                for genre in genres:
                    if genre not in games_by_genre:
                        games_by_genre[genre] = []
                    games_by_genre[genre].append(app_name)

        total_playtime_by_genre = {}  # Diccionario para almacenar el tiempo total de juego por género

        for i in range(len(g_list)):
            genre = g_list[i]
            if genre in games_by_genre:
                total_time = 0  # Inicializar el tiempo total en cero para este género
                for game in games_by_genre[genre]:
                    if game in total_playtime:
                        total_time += total_playtime[game]
                total_playtime_by_genre[genre] = total_time

        sorted_genres_by_playtime = sorted(total_playtime_by_genre.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(len(sorted_genres_by_playtime)):
            if sorted_genres_by_playtime[i][0] == genero:
                return{f'El género {genero} se encuentra en el puesto número' : i+1}
    else:
        return{'El género brindado no se encuentra en la base de datos, por favor revíselo'}


@app.get("/userforgenre/{genero}")
def userforgenre(genero: str):
    g_list = genres_list()
    if genero in g_list:
        user_games_playtime = {}
        games_by_genre = {}

        for i in range(len(users_items['user_id'])):
            user_id = users_items['user_id'][i]
            for item in users_items['items'][i]:
                if 'playtime_forever' in item:
                    game_name = item['item_name']
                    playtime_forever = int(item['playtime_forever'])
                    
                    if user_id not in user_games_playtime:
                        user_games_playtime[user_id] = []
                    user_games_playtime[user_id].append((game_name, playtime_forever))

        for i in range(len(steam_games)):
            app_name = steam_games['app_name'][i]
            genres = steam_games['genres'][i]
                
            if genres is not None:
                for genre in genres:
                    if genre not in games_by_genre:
                        games_by_genre[genre] = []
                    games_by_genre[genre].append(app_name)

        target_genre_games = set(games_by_genre.get(genero, []))

        genre_user_playtime = []
        for user_id, games_playtime in user_games_playtime.items():
            user_genre_playtime = 0
            for game_name, playtime_forever in games_playtime:
                if game_name in target_genre_games:
                    user_genre_playtime += playtime_forever
            genre_user_playtime.append((user_id, user_genre_playtime))
        
        # Ordenar la lista en función del tiempo de juego y obtener los 5 usuarios con más playtime
        best_gamers = sorted(genre_user_playtime, key=lambda x: x[1], reverse=True)[:5]

        user_ids = [user_id for user_id, _ in best_gamers]
        user_urls = [users_items['user_url'][users_items['user_id'] == user_id].iloc[0] for user_id in user_ids]

        result_dict = {'user_id': user_ids, 'user_url': user_urls}

        return result_dict
    else:
        return {'El género brindado no se encuentra en la base de datos, por favor revíselo'}
    
@app.get("/developer/{desarrollador}")
def developer(desarrollador: str):
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
def sentiment_analysis( anio : int ): 
    if anio > 2009 and anio < 2016:
        positivo = 0
        negativo = 0
        neutral = 0
        for i in range(len(user_reviews)):
            for j in range(len(user_reviews['reviews'][i])):
                if user_reviews['reviews'][i][j]['posted'] is None:
                    pass
                if user_reviews['reviews'][i][j]['posted'] is not None:
                    if int(user_reviews['reviews'][i][j]['posted'][:4]) == anio:
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