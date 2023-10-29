import pandas as pd
import fastapi
from sklearn.metrics.pairwise import cosine_similarity

app = fastapi.FastAPI()

df_items = pd.read_csv('Datasets/items_reduc.csv', encoding='utf-8', sep = '|')
df_reviews = pd.read_csv('Datasets/reviews_sa.csv')
df_steam_exploded = pd.read_csv('Datasets/steam_exploded.csv')
df_steam= pd.read_csv('Datasets/steam_games.csv')



@app.get('/')
async def home():
    return {'Data' : 'Testing'}

@app.get('/developer/{desarrollador}')
async def developer(desarrollador: str):
    # Convertir el nombre del desarrollador proporcionado a minúsculas
    desarrollador_lower = desarrollador.lower()

    # Filtrar los dataframes para el desarrollador proporcionado
    # Convertir la columna 'developer' a minúsculas antes de la comparación
    developer_games = df_steam[df_steam['developer'].str.lower() == desarrollador_lower]

    # Si el desarrollador no tiene juegos en el dataframe
    if developer_games.empty:
        return 'El desarrollador no tiene juegos en la plataforma'

    # Obtener los años únicos de los juegos del desarrollador
    years = developer_games['year'].unique()

    result = {}

    for year in years:
        # Filtrar los juegos del desarrollador para el año especificado
        developer_games_year = developer_games[developer_games['year'] == year]

        # Contar la cantidad de items
        item_count = len(developer_games_year)

        # Calcular el porcentaje de contenido gratuito
        free_games = developer_games_year[developer_games_year['price'] == 0]
        free_percentage = (len(free_games) / item_count) * 100

        result[year] = {'Cantidad de Items': item_count, 'Contenido Free': f'{free_percentage}%'}

    return {desarrollador: result}


@app.get('/userdata/{User_id}')
async def userdata(User_id: str):
    # Filtrar los dataframes para el usuario proporcionado
    # Obtener las filas del dataframe df_items donde la columna 'user_id' coincide con User_id
    user_items = df_items[df_items['user_id'] == User_id]

    # Obtener las filas del dataframe df_reviews donde la columna 'user_id' coincide con User_id
    user_reviews = df_reviews[df_reviews['user_id'] == User_id]

    # Calcular el dinero gastado
    # Obtener los ids de los items que el usuario ha comprado
    item_ids = user_items['item_id'].unique()

    # Calcula la suma de los precios de los items comprados por el usuario
    steam_prices = df_steam[df_steam['item_id'].isin(item_ids)]['price']
    money_spent = steam_prices.sum()

    # Calcular el porcentaje de recomendación
    total_reviews = len(user_reviews)
    if total_reviews > 0:
        recommended_reviews = user_reviews[user_reviews['recommend']]['recommend'].count()
        recommendation_percentage = (recommended_reviews / total_reviews) * 100
    else:
        recommendation_percentage = 0

    # Contar la cantidad de items
    item_count = len(item_ids)

    return {"Usuario": User_id, "Dinero gastado": f"{money_spent} USD", "% de recomendación": f"{recommendation_percentage}%", "cantidad de items": item_count}


@app.get('/UserForGenre/{genero}')
async def UserForGenre(genero: str):
    # Filtrar juegos del género especificado
    gf = df_steam_exploded[df_steam_exploded['genres'] == genero]

    genero_erroneo = """
Vuelva a ingresar el género y verifique que sea uno de los siguientes, respetando mayúsculas y puntuación:

1. Indie
2. Action
3. Casual
4. Adventure
5. Strategy
6. Simulation
7. RPG
8. Free to Play
9. Early Access
10. Sports
11. Massively Multiplayer
12. Racing
14. Utilities
15. Web Publishing
16. Education
17. Video Production
18. Software Training
19. Audio Production
20. Photo Editing
21. Design &amp; Illustration
"""
    if gf.empty or genero ==  'Accounting':
        return genero_erroneo

    # Realizar un left join para mantener todas las filas de gf y luego eliminar filas con playtime_forever NaN
    m = gf.merge(df_items, on=['item_name'], how='left').dropna(subset=['playtime_forever']).drop_duplicates()

    # Convertir la columna 'release_date' a tipo datetime
    m['release_date'] = pd.to_datetime(m['release_date'], errors='coerce')

    # Agrupar por usuario y calcular el tiempo total de juego por usuario
    user_playtime = m.groupby('user_id')['playtime_forever'].sum()

    # Encontrar el usuario con más horas jugadas en ese género
    max_playtime_user = user_playtime.idxmax()

    # Filtrar el DataFrame para el usuario con más horas jugadas
    user_max_playtime = m[m['user_id'] == max_playtime_user]

    # Calcular la acumulación de horas jugadas por año para ese usuario
    year_playtime = user_max_playtime.groupby(user_max_playtime['release_date'].dt.year)['playtime_forever'].sum()

    # Crear la lista de acumulación de horas jugadas por año en el formato especificado
    horas_por_anio = [{"Año": int(year), "Horas": int(hours)} for year, hours in year_playtime.items()]

    return {"Usuario con más horas jugadas para el género " + genero: max_playtime_user, "Horas jugadas por año": horas_por_anio}

@app.get('/best_developer_year/{anio}')
async def best_developer_year(anio: int):
    # Filtrar los dataframes para el año especificado
    reviews_year = df_reviews[df_reviews['year_posted'] == str(anio)]
    steam_year = df_steam[df_steam['year'] == str(anio)]

    # Filtrar las reseñas para quedarse solo con las recomendadas
    reviews_recommended = reviews_year[reviews_year['recommend'] == True]

    # Realizar un merge para combinar la información de las reseñas y juegos
    merged = reviews_recommended.merge(steam_year, on='item_id')

    # Agrupar por desarrollador y contar el número de reseñas recomendadas
    developer_count = merged.groupby('developer')['recommend'].count()

    # Ordenar los desarrolladores por el número de reseñas recomendadas de forma descendente
    sorted_developers = developer_count.sort_values(ascending=False)

    # Seleccionar los primeros tres desarrolladores
    top_3_developers = sorted_developers.head(3)

    # Crear el resultado en el formato especificado
    result = {str(anio): top_3_developers.to_dict()}

    return result


@app.get('/eveloper_reviews_analysis/{desarrolladora}')
async def developer_reviews_analysis(desarrolladora: str):
    # Hacer un merge entre df_reviews_sa y df_steam usando la columna 'item_id'
    df_merged = pd.merge(df_reviews, df_steam, on='item_id')

    # Filtrar los juegos desarrollados por la desarrolladora dada
    df_filtered = df_merged[df_merged['developer'] == desarrolladora]

    # Filtrar las reseñas con análisis de sentimiento positivo o negativo
    df_filtered = df_filtered[df_filtered['sentiment_analysis'].isin([0, 2])]

    # Contar la cantidad de reseñas negativas y positivas
    negative_count = df_filtered[df_filtered['sentiment_analysis'] == 0].shape[0]
    positive_count = df_filtered[df_filtered['sentiment_analysis'] == 2].shape[0]

    # Crear el diccionario de retorno
    result = {desarrolladora: {'Negative': negative_count, 'Positive': positive_count}}

    return result


from sklearn.metrics.pairwise import cosine_similarity


df_reviews_shuffled = df_reviews.head(10000)
df_reviews_shuffled = df_reviews_shuffled.sample(frac=1, random_state=42)

# Crear una matriz de usuarios como características y juegos como filas
user_item_matrix = df_reviews_shuffled.pivot_table(index='user_id', columns='item_id', values='sentiment_analysis').fillna(0)

# Calcular la similitud entre usuarios usando la similitud del coseno
user_similarity = cosine_similarity(user_item_matrix)


@app.get('/recomendacion_usuario/{user_id}')
async def recomendacion_usuario(user_id):
    # Obtener la fila correspondiente al usuario ingresado
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

    # Calcular la similitud entre el usuario ingresado y todos los demás usuarios
    similarities = cosine_similarity(user_vector, user_item_matrix)

    # Obtener los juegos que los usuarios similares a 'user_id' han disfrutado
    user_reviews = user_item_matrix.loc[user_id]
    similar_users = user_item_matrix.index[similarities.argsort()[0][-6:-1]]
    recommended_items = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)

    # Filtrar los juegos que el usuario ya ha jugado
    recommended_items = recommended_items[~recommended_items.index.isin(user_reviews[user_reviews > 0].index)]

    return recommended_items.index.tolist()[:5]

