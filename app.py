import numpy as np
import pickle

from flask import Flask, request, render_template
app = Flask(__name__, template_folder='template')

s_y = pickle.load(open('svr_pkl_sy.pkl','rb'))

s_x = pickle.load(open('svr_pkl_sx.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))
model_svr = pickle.load(open('svr_pkl.pkl', 'rb'))
model_dtr = pickle.load(open('dtr_pkl.pkl','rb'))

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/fish_page', methods=['GET','POST'])
def fish_page():
    return render_template('fish_page.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'age' in request.form:
        year_age = request.form['age']
        year_age = float(year_age)
        if int(year_age) < 10:
            ff = [[np.array(year_age)]]
            predicted_val = model.predict(ff)
            output = round(predicted_val[0], 2)
            return render_template('predict.html', prediction_text='Length of Fish: {} cm'.format(output))
        else:
            return render_template('predict.html', prediction_text='Invalid Age')
    else:
        return render_template('predict.html', prediction_text=' ')


@app.route('/svr', methods=['GET', 'POST'])
def svr():
    if 'age' in request.form:
        year_age = request.form['age']
        year_age = float(year_age)
        if int(year_age) < 10:
            ff = [[np.array(year_age)]]
            predicted_val = model_dtr.predict(ff)
            output = round(predicted_val[0], 2)
            return render_template('svr.html', prediction_text='Length of Fish: {} cm'.format(output))
        else:
            return render_template('svr.html', prediction_text='Invalid Age')
    else:
        return render_template('svr.html', prediction_text=' ')


@app.route('/dtr', methods=['GET','POST'])
def dtr():
    if 'age' in request.form:
        year_age = request.form['age']
        year_age = float(year_age)
        if int(year_age) < 10:
            ff = [[np.array(year_age)]]
            predicted_val = s_y.inverse_transform(model_svr.predict(s_x.transform(ff)))
            output = round(predicted_val[0], 2)
            return render_template('dtr.html', prediction_text='Length of Fish: {} cm'.format(output))
        else:
            return render_template('dtr.html', prediction_text='Invalid Age')
    else:
        return render_template('dtr.html', prediction_text=' ')


@app.route('/rtr_fish', methods=['GET','POST'])
def rtr_fish():
    if 'age' in request.form:
        year_age = request.form['age']
        year_age = float(year_age)
        if int(year_age) < 10:
            ff = [[np.array(year_age)]]
            predicted_val = s_y.inverse_transform(model_svr.predict(s_x.transform(ff)))
            output = round(predicted_val[0], 2)
            return render_template('rtr_fish.html', prediction_text='Length of Fish: {} cm'.format(output))
        else:
            return render_template('rtr_fish.html', prediction_text='Invalid Age')
    else:
        return render_template('rtr_fish.html', prediction_text=' ')

if __name__ == "__main__":
    app.run(debug=True, port=8000)