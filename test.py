from predict import Predictor

predictor = Predictor()
predictor.setup()

# Menjalankan prediksi
# Pass max_length as a keyword argument
output = predictor.predict(prompt="Panggil Aku Ganteng", max_length=50)  
print(output)