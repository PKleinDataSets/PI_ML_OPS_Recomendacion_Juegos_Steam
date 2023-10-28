import pandas as pd
import fastapi
from sklearn.metrics.pairwise import cosine_similarity

app = fastapi.FastAPI()

df_items= pd.read_csv('Datasets/items.csv')
df_reviews = pd.read_csv('Datasets/reviews_sa.csv')
df_steam_exploded = pd.read_csv('Datasets/steam_exploded.csv')
df_steam= pd.read_csv('Datasets/steam_games.csv')



@app.get('/')
async def home():
    return {'Data' : 'Testing'}

@app.get('/PlayTimeGenre/{genero}')
async def PlayTimeGenre(genero: str):
    # Filtrar juegos del género especificado
    gf = df_steam_exploded[df_steam_exploded['genres'] == genero]

    if gf.empty:
        return {"No se encontraron juegos para el género especificado": None}

    # Realizar un left join para mantener todas las filas de gf y luego eliminar filas con playtime_forever NaN
    m = gf.merge(df_items, on=['item_name'], how='left').dropna(subset=['playtime_forever']).drop_duplicates()

    # Convertir la columna 'release_date' a tipo datetime
    m['release_date'] = pd.to_datetime(m['release_date'], errors='coerce')

    # Agrupar por año de lanzamiento y calcular el tiempo total de juego por año
    year_playtime = m.groupby(m['release_date'].dt.year)['playtime_forever'].sum()

    # Encontrar el año con más horas jugadas
    max_playtime_year = year_playtime.idxmax()

    return {"Año de lanzamiento con más horas jugadas para el género " + genero: int(max_playtime_year)}


@app.get('/UserForGenre/{genero}')
async def UserForGenre(genero: str):
    # Filtrar juegos del género especificado
    gf = df_steam_exploded[df_steam_exploded['genres'] == genero]

    if gf.empty:
        return {"No se encontraron juegos para el género especificado": None, "Horas jugadas por año": {}}

    # Realizar un left join para mantener todas las filas de gf y luego eliminar filas con playtime_forever NaN
    m = gf.merge(df_items, on=['item_name'], how='left').dropna(subset=['playtime_forever']).drop_duplicates()

    # Convertir la columna 'release_date' a tipo datetime
    m['release_date'] = pd.to_datetime(m['release_date'], errors='coerce')

    # Agrupar por usuario y calcular el tiempo total de juego por usuario
    user_playtime = m.groupby('user_id_y')['playtime_forever'].sum()

    # Encontrar el usuario con más horas jugadas en ese género
    max_playtime_user = user_playtime.idxmax()

    # Filtrar el DataFrame para el usuario con más horas jugadas
    user_max_playtime = m[m['user_id_y'] == max_playtime_user]

    # Calcular la acumulación de horas jugadas por año para ese usuario
    year_playtime = user_max_playtime.groupby(user_max_playtime['release_date'].dt.year)['playtime_forever'].sum()

    # Crear la lista de acumulación de horas jugadas por año en el formato especificado
    horas_por_anio = [{"Año": int(year), "Horas": int(hours)} for year, hours in year_playtime.items()]

    return {"Usuario con más horas jugadas para el género " + genero: max_playtime_user, "Horas jugadas por año": horas_por_anio}

@app.get('/UsersRecommend/{anio}')
async def UsersRecommend(anio: int):
    # Filtrar las revisiones para el año dado y donde recommend sea True y el sentimiento sea positivo o neutral
    reviews_filtered = df_reviews[(df_reviews['posted'].str[-4:] == str(anio)) & \
     (df_reviews['recommend'] == True) & (df_reviews['sentiment_analysis']> 0)]#.isin([1, 2]))]

    # Realizar un merge para obtener los nombres de los juegos
    merged_reviews = reviews_filtered.merge(df_steam[['item_id', 'item_name']], on='item_id', how='left')


    # Agrupar por item_name y contar las revisiones para cada juego
    game_counts = merged_reviews.groupby('item_name')['recommend'].count().reset_index()

    # Ordenar los juegos por la cantidad de revisiones en orden descendente
    top_games = game_counts.sort_values(by='recommend', ascending=False)

    # Tomar los 3 juegos principales
    top_3_games = top_games.head(3)

    # Crear la lista de retorno con el formato especificado
    result = [{"Puesto {}: ".format(i + 1): game} for i, game in enumerate(top_3_games['item_name'])]

    return result

@app.get('/UsersNotRecommend/{anio}')
async def UsersNotRecommend(anio: int):
    # Filtrar las revisiones para el año dado, donde recommend sea False y el sentimiento sea negativo
    reviews_filtered = df_reviews[(df_reviews['posted'].str[-4:] == str(anio)) & \
        (df_reviews['recommend'] == False) & (df_reviews['sentiment_analysis'] == 0)]
    
    # Realizar un merge para obtener los nombres de los juegos
    merged_reviews = reviews_filtered.merge(df_steam[['item_id', 'item_name']], on='item_id', how='left')
    
    # Agrupar por item_name y contar las revisiones para cada juego
    game_counts = merged_reviews.groupby('item_name')['recommend'].count().reset_index()
    
    # Ordenar los juegos por la cantidad de revisiones en orden descendente
    bottom_games = game_counts.sort_values(by='recommend', ascending=True)
    
    # Tomar los 3 juegos menos recomendados
    bottom_3_games = bottom_games.head(3)
    
    # Crear la lista de retorno con el formato especificado
    result = [{"Puesto {}: ".format(i + 1): game} for i, game in enumerate(bottom_3_games['item_name'])]

    return result


@app.get('/sentiment_analysis/{anio}')
async def sentiment_analysis(anio: int):
    # Filtrar las revisiones para el año de lanzamiento dado
    reviews_filtered = df_steam[df_steam['release_date'].str[:4] == str(anio)]

    # Realizar un merge entre las revisiones filtradas y df_reviews para obtener el análisis de sentimiento
    merged_reviews = reviews_filtered.merge(df_reviews[['item_id', 'sentiment_analysis']], on='item_id', how='left')

    # Contar las revisiones con análisis de sentimiento y agrupar por categoría de sentimiento
    sentiment_counts = merged_reviews['sentiment_analysis'].value_counts()

    # Crear un diccionario con las categorías de sentimiento y sus conteos
    result = {}
    for sentiment_category, count in sentiment_counts.items():
        result[str(sentiment_category)] = count

    # Verificar si no hay reseñas para el año y establecer todos los valores en 0
    if not result:
        result = {'0.0': 0, '1.0': 0, '2.0': 0}

    new_result = {'Negative': result.get('0.0', 0), 'Neutral': result.get('1.0', 0), 'Positive': result.get('2.0', 0)}

    return new_result


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
    similar_users = df_reviews['user_id'][similarities.argsort()[0][-6:-1]]
    recommended_items = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
    
    # Filtrar los juegos que el usuario ya ha jugado
    recommended_items = recommended_items[recommended_items.index.isin(user_reviews[user_reviews == 0].index)]
    
    return recommended_items.index.tolist()[:5]











