from flask import Flask, request,\
        render_template, flash, redirect, make_response, send_file, session
import sys
sys.path.append('../code')
import cPickle as pickle
import pandas as pd

from model import Model

app = Flask(__name__)


@app.route('/form', methods=['GET', 'POST'])
def submission_page():
    dict_agency = {'dod' : 'Department of Defense',
               'dhs' : 'Department of Homeland Security',
               'dt'  : 'Department of Transportation',
               'dc'  : 'Department of Commerce',
               'ded' : 'Department of Education',
               'den' : 'Department of Energy',
               'epa' : 'Environmental Protection Agency',
               'nasa': 'National Aeronautics and Space Administration',
               'dhhs': 'Department of Health and Human Services',
               'nsf' : 'National Science Foundation',
               'da'  : 'Department of Agriculture'}

    list_ownership = [(1,'Hubzone Owned'),
                      (2,'Socially and Economically Disadvantaged'),
                      (3,'Woman Owned')]
    return render_template('start_form.html', agencies=dict_agency.values(),
                            ownership=list_ownership)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    #getting all the information from the form
    session['Company Name'] = request.form['Company Name']
    session["Solicitation Year"] = request.form["Solicitation Year"]
    session["Award Amount"] = request.form["Award Amount"]
    session["Agency"] = request.form["Agency"]
    session["Abstract"] = request.form["Abstract"]
    session["Employees"] = request.form["Employees"]
    try:
        session["1"]=request.form['1']
    except:
        session["1"]='off'
    try:
        session["2"]=request.form['2']
    except:
        session["2"]='off'
    try:
        session["3"]=request.form['3']
    except:
        session["3"]='off'

    for i in ["1", "2", "3"]:
        if session[i] == 'off':
            session[i] = 'No'
        else:
            session[i] = 'Yes'

    #making a dataframe with it:
    input_df = pd.DataFrame({u'index': 'not used',
                             u'Company': session['Company Name'],
                             u'Award Title': 'not used',
                             u'Agency': session["Agency"],
                             u'Branch': 'not used',
                             u'Phase': 'not used',
                             u'Program': 'not used',
                             u'Agency Tracking #': 'not used',
                             u'Contract': 'not used',
                             u'Award Start Date': 'not used',
                             u'Award Close Date': 'not used',
                             u'Solicitation #': 'not used',
                             u'Solicitation Year': session['Solicitation Year'],
                             u'Topic Code': 'not used',
                             u'Award Year': 'not used',
                             u'Award Amount': session['Award Amount'],
                             u'DUNS': 'not used',
                             u'Hubzone Owned': session["1"],
                             u'Socially and Economically Disadvantaged': session["2"],
                             u'Woman Owned': session["3"],
                             u'# Employees': session["Employees"],
                             u'Company Website': 'not used',
                             u'Address1': 'not used',
                             u'Address2': 'not used',
                             u'City': 'not used',
                             u'State': 'not used',
                             u'Zip': 'not used',
                             u'Contact Name': 'not used',
                             u'Contact Title': 'not used',
                             u'Contact Phone': 'not used',
                             u'Contact Email': 'not used',
                             u'PI Name': 'not used',
                             u'PI Title': 'not used',
                             u'PI Phone': 'not used',
                             u'PI Email': 'not used',
                             u'RI Name': 'not used',
                             u'RI POC Name': 'not used',
                             u'RI POC Phone': 'not used',
                             u'Research Keywords': 'not used',
                             u'Abstract': session["Abstract"],
                             u'to_phase_II': 'not used'})

    model_form = Model(input_df, [0], [0])
    model_form.process_text('Abstract')
    model_form.prepare_LogReg()

    return render_template('predictions.html',
                            name=session['Company Name'],
                            year=session['Solicitation Year'],
                            award=session['Award Amount'],
                            agency=session["Agency"],
                            abstract=session["Abstract"],
                            employee=session["Employees"],
                            owner1=session["1"],
                            owner2=session["2"],
                            owner3=session["3"])

if __name__ == '__main__':

    model = pickle.load(open('data/model.pkl', 'rb'))
    scaler = pickle.load(open('data/scaler.pkl', 'rb'))

    app.secret_key = 'super secret key'
    app.run(host='0.0.0.0', port=8080, debug=True)
