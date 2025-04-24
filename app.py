from flask import Flask, request, render_template
import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        # Collect and convert form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Create DataFrame from input
        input_df = data.get_data_as_data_frame()
        logging.info("Input DataFrame: %s", input_df)

        # Make prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)
        logging.info("Prediction Result: %s", prediction)

        return render_template('home.html', results=prediction[0])
    
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return render_template('home.html', results="Error occurred during prediction. Please check input values.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
