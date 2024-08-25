from flask import Flask, jsonify, request
import joblib
import sklearn

app = Flask(__name__)

@app.route('/')
def root():
    return "Home"
@app.route('/prediccion', methods=['GET'])
def prediccion():
    # Obtener los parámetros de la URL
  
    Irradiancia_solar = float(request.args.get('Irradiancia_solar'))
    Temperatura_max = float(request.args.get('Temperatura_max'))
    Precipitaciones_mm = float(request.args.get('Precipitaciones_mm'))
    Humedad_Relativa = float(request.args.get('Humedad_Relativa'))
    Velocidad_viento = float(request.args.get('Velocidad_viento'))
    Velocidad_viento_max = float(request.args.get('Velocidad_viento_max'))
    
    
    # Crear un diccionario con los parámetros
    model = joblib.load('modelRandomForest.pkl')
    resultado = model.predict([[Irradiancia_solar, Temperatura_max, Precipitaciones_mm, Humedad_Relativa, Velocidad_viento,Velocidad_viento_max]]).tolist()
    
    # Crear un diccionario con los parámetros
    params_dict = {
        'resultado': resultado[0],
        'sklearn':sklearn.__version__
    }

    # Devolver el diccionario como JSON
    return jsonify(params_dict)
if __name__ == '__main__':
    app.run(debug=True)
