from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Teams
teams = [
    'Sunrisers Hyderabad', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Delhi Capitals',
    'Mumbai Indians', 'Chennai Super Kings', 'Rajasthan Royals'
]

# Venues
venues = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load model + transformer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models.pkl", "rb") as f:
    models = pickle.load(f)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', teams=teams, venues=venues, result=None)


@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form.get('batting_team')
    bowling_team = request.form.get('bowling_team')
    venue = request.form.get('venue')

    wickets = int(request.form.get('wickets'))
    overs = int(request.form.get('overs'))
    target = int(request.form.get('target'))
    score = int(request.form.get('score'))

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    delivery_left = 126 - (overs * 6 + (6 - (balls_left % 6)))  # optional tweak

    # Create dataframe
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [venue],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'delivery_left': [delivery_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

   

    # Predict probabilities
    proba = model.predict_proba(input_df)[0]

    # Convert to percentages
    loss = round(float(proba[0]) * 100, 2)
    win = round(float(proba[1]) * 100, 2)

    result = {
        'loss': loss,
        'win': win
    }

    return render_template(
        'index.html',
        teams=teams,
        venues=venues,
        result=result
    )


if __name__ == '__main__':
    app.run(debug=True)
