from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
from wtforms import Form, FloatField, validators, RadioField
from wtforms.widgets.html5 import RangeInput
import json
import jsonpickle

import tensorflow as tf
import numpy as np
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import  Optimizer
import matplotlib.pyplot as plt
import random

from data_handling import denormalise_batch
from soundgen_tf import vec_to_freqpower, calc_rms_sines, get_freqs_powers
from gSheets import Sheets_IO
from user_data import get_random_string


MAX_TRIALS = 30

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)

app.config['SECRET_KEY'] = 'This_is_the_secret_key_12243523225907662'

SAMPLE_SPREADSHEET_ID = '1IW3prWC6neaXqvNfUXJcHqtuxS3DFIlB5FvFoyqXxXI'
#SAMPLE_SPREADSHEET_ID = '1P_wU0rU6_rdw0WBomkZm8Ofbkv4AG96N8hmvMo2gXKc'
#SAMPLE_SPREADSHEET_ID = '1JGRd59WRZ7nr6cgwKJz_IKicDfZutj2UtcjHkC5IOBk'
results_sheet = Sheets_IO(SAMPLE_SPREADSHEET_ID)


# Model
class InputForm(Form):
    r = FloatField(validators=[validators.InputRequired()])

class CalibrationForm(Form):
    radio_group = RadioField('label', choices=[('headphones','Headphones'),('loudspeakers','Loudspeakers')], validators=[validators.InputRequired()])
    f = FloatField(validators=[validators.InputRequired()])

#decoder = tf.keras.models.load_model('models/decoder_4_params_fitted')
decoder = tf.keras.models.load_model('models/master_VAE/VAE_sin_parameters_4_higher_kl/decoder')
#x_batch_mean = np.array([1.52481007e+03, 1.50019014e+00, 3.25621367e-02, 5.00145407e-01])
#x_batch_std = np.array([8.51702500e+02, 1.11811500e+00, 5.72985211e+01, 2.88592207e-01])
x_batch_mean = np.array([1.52404555e+03, 1.50050163e+00, 7.55882263e-03, 4.99974079e-01])
x_batch_std = np.array([8.51562896e+02, 1.11773102e+00, 5.73145759e+01, 2.88692592e-01])
range_val = 3
space = [Real(-range_val, range_val, name='X'),
             Real(-range_val, range_val, name='Y')]

bigger_space = [Real(0, 3000, name='fund_freq'),
                Integer(0, 4, name='n_harmonics'),
                Integer(-99, 100, name='detuning_hz'),
                Real(0, 1, name='harmonics_power')]

vals_list=[]

class TrialNum(object):

    def __init__(self):
        self.trial_num =0

    def inc(self):
        self.trial_num=self.trial_num+1
        return self.trial_num

    def get_num(self):
        return self.trial_num

'''
def lowdim_trial_number():
    if 'lowdim_trial' in session:
        session['lowdim_trial'] = session.get('lowdim_trial') + 1  # reading and updating session data
        session.modified = True
    else:
        session['lowdim_trial'] = 1 # setting session data
        session.modified = True
    return session.get('lowdim_trial')

def highdim_trial_number():
    if 'highdim_trial' in session:
        session['highdim_trial'] = session.get('highdim_trial') + 1  # reading and updating session data
    else:
        session['highdimm_trial'] = 1 # setting session data
    return session.get('highdim_trial')
'''




results_sheet = Sheets_IO(SAMPLE_SPREADSHEET_ID)

num_responses = 0

test_templates = [ #'compare_sounds.html']
                  #'compare_sounds_inverse.html']
                'compare_sounds_responsive.html']
test_template = random.choice(test_templates)

print('we will be using ' + test_template)


def get_synth_params(coords):
    '''take [x,y] corrdinate and return synth parameters.'''

    params_normalised = decoder.predict([coords])
    dim4_params = denormalise_batch(params_normalised, x_batch_std, x_batch_mean).numpy()

    rounding_vals = np.array([0, 1, 2, 3])
    diffs = abs(rounding_vals - dim4_params[0, 1])
    closest_index = np.argmin(diffs)
    dim4_params[0, 1] = rounding_vals[closest_index]

    params = vec_to_freqpower(dim4_params).numpy()
    return params


def get_adjusted_f_val(val, rms, noise_level):
    '''
    Adjusts the tone 'score' based on the RMS level of the noise and the tone.
    High noise levels will reduce the score, as will low rms.
    A lower 'score' equates to a tone being more audible in noise.
    :param val: The value returned from the listening test
    :param rms: The rms value of the tone which was presented
    :param noise_level: The value of the noise present when listening.
    :return: corrected noise level
    '''

    return (val*rms)/noise_level


def session_optimizer():
    if 'opt' in session:
        return session.get('opt')
    else:
        session['opt'] = jsonpickle.encode(Optimizer([(-3.0, 3.0), (-3.0, 3.0)], "GP", acq_func="EI",
                        acq_optimizer="sampling",
                        initial_point_generator="lhs"))
        return session.get('opt')


def session_big_optimizer():
    if 'big_opt' in session:
        return session.get('big_opt')
    else:
        session['big_opt'] = jsonpickle.encode(Optimizer([(0, 3000), (0, 4), (-99, 100), (0.0, 1.0)], "GP", acq_func="EI",
                acq_optimizer="sampling",
                initial_point_generator="lhs"))
        return session.get('big_opt')

def session_n_trials(n_trials=30):
    if 'n_trials' in session:
        return session.get('n_trials')
    else:
        session['n_trials'] = n_trials
        print('setting the number of trials to ', n_trials)
        return session.get('n_trials')

def session_noise_vols_low():
    if 'noise_vols_low' in session:
        return session.get('noise_vols_low')
    else:
        session['noise_vols_low'] = []
        session.modified = True
        print('creating list or range of noise volumes for low dimension')
        return session.get('noise_vols_low')

def session_noise_vols_high():
    if 'noise_vols_high' in session:
        return session.get('noise_vols_high')
    else:
        session['noise_vols_high'] = []
        session.modified = True
        print('creating list or range of noise volumes for high dimension')
        return session.get('noise_vols_high')

def session_user_ID():
    if 'user_ID' in session:
        return session.get('user_ID')
    else:
        session['user_ID'] = get_random_string(12)
        return session.get('user_ID')

def get_validation_vals():
    if 'validation_vals' in session:
        return session.get('validation_vals')
    else:
        session['validation_vals'] = validate_vals()
        return session.get('validation_vals')

def session_validation_results():
    if 'validation_results' in session:
        return session.get('validation_results')
    else:
        session['validation_results'] = []
        return session.get('validation_results')

def session_master_vol(vol=1):
    if 'master_vol' in session:
        return session.get('master_vol')
    else:
        session['master_vol'] = vol
        return session.get('master_vol')

def session_device(device='unspecified'):
    if 'device' in session:
        return session.get('device')
    else:
        session['device']=device
        return session.get('device')

def get_noise_vol():
    return (np.random.rand() / 4.0) + 0.75

def get_slider_range():
    return random.randint(2, 5)


def validate_vals():
    '''
    Get the lowest value from each row of the spreadsheet and use it to generate a sample for testing.
    :return:
    '''

    #Get list of points and minima from each trial:
    #lowdim minima
    minvals_low = []
    minvals_high = []
    lowvals=[]


    n_results = 30
    BATCH_SHEET_RANGE = []

    #low dimension
    BATCH_SHEET_RANGE = []
    for n in range(0, n_results):
        BATCH_SHEET_RANGE.append(results_sheet.get_row_arg('Sheet1', n * 6 + 6))
        BATCH_SHEET_RANGE.append(results_sheet.get_row_arg('Sheet1', n * 6 + 3, n * 6 + 4))

    low_dim_res = results_sheet.get_batch_rows(BATCH_SHEET_RANGE) # all the results. even rows y values, odd rows X (2dims).
    #organise the reslts. Every 1 row is val (y), 2 row is X(2 dimensions)
    for i in range(0, int(len(low_dim_res)/2)):
        if 'values' in low_dim_res[2*i]:
            tempdict = {}
            tempdict['y_elem'] = low_dim_res[2*i]['values']
            tempdict['X_elem'] = low_dim_res[2*i+1]['values']
            min_value_index = np.argmin(tempdict['y_elem'])
            minvals_low.append([float(tempdict['X_elem'][0][min_value_index]), float(tempdict['X_elem'][1][min_value_index])])

    #right, now take all the values, and convert to format for synth

    low_decoded = []
    for item in minvals_low:
        low_decoded.append(np.squeeze(np.abs(get_synth_params(item))).tolist())
        #get_synth_params(next_x)
        #vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()

    #High dimension:
    BATCH_SHEET_RANGE = []
    for n in range(0, n_results):
        BATCH_SHEET_RANGE.append(results_sheet.get_row_arg('Sheet2', n * 8 + 8))
        BATCH_SHEET_RANGE.append(results_sheet.get_row_arg('Sheet2', n * 8 + 3, n * 8 + 6))
    high_dim_res = results_sheet.get_batch_rows(BATCH_SHEET_RANGE)
    # organise the reslts. Every 1 row is val (y), 2nd row is X(4 dimensions)
    for i in range(0, int(len(high_dim_res)/2)):
        if 'values' in high_dim_res[2 * i]:
            tempdict = {}
            tempdict['y_elem'] = high_dim_res[2 * i]['values']  # every otehr row
            tempdict['X_elem'] = high_dim_res[2 * i + 1]['values']
            min_value_index = np.argmin(tempdict['y_elem'])
            minvals_high.append([float(tempdict['X_elem'][0][min_value_index]), float(tempdict['X_elem'][1][min_value_index]), float(tempdict['X_elem'][2][min_value_index]), float(tempdict['X_elem'][3][min_value_index])])

    high_decoded = []
    for item in minvals_high:
        high_decoded.append(get_freqs_powers([item]))

    all_vals_list =[]

    for item in high_decoded:
        vals = np.squeeze(tf.concat(item, axis=1).numpy()).tolist()


        tempdict={'values': vals, 'type': 'high'}

        all_vals_list.append(tempdict)

    for item in low_decoded:


        tempdict = {'values': item, 'type': 'low'}

        all_vals_list.append(tempdict)

    random.shuffle(all_vals_list)

    return all_vals_list


@app.route('/start')
@app.route('/')
def start():
    session.clear()
    session_user_ID()
    print('*********USER STARTING WITH ID ' + str(session_user_ID()) + ' *************')
    with open("static/introText", "r") as f:
        content = f.read()


    n_trials_range = [10, 30, 50]
    test_order = np.random.permutation(['highdim', 'lowdim'])
    session_n_trials(random.choice(n_trials_range))
    #session_n_trials(5)
    session_first_test = test_order[0]
    session_second_test = test_order[1]
    session['test_order']=test_order
    session['lowdim_complete'] = False
    session['highdim_complete'] = False
    session.modified = True

    calibration_page='calibrate'



    return render_template("startpage.html", content=content, next_page=calibration_page)


@app.route('/calibrate',  methods=['POST', 'GET'])
def calibrate():
    form = InputForm(request.form)
    calib_form = CalibrationForm(request.form)


    if request.method == 'POST' and calib_form.validate():
        #Save the master volume level, and start the test.
        f = calib_form.f.data
        session_master_vol(f) #save to session
        session_device(calib_form.data['radio_group']) #save the device
        session.modified = True

        #go to the first stage in the test:
        return redirect(url_for(session['test_order'][0]))



    else:
        return render_template("calibrate.html", form=calib_form)


@app.route('/pause')
def pause():
    if (session['lowdim_complete'] is True) and (session['highdim_complete'] is True): #both tests have been done, save results


        opt_big = jsonpickle.decode(session_big_optimizer())
        opt = jsonpickle.decode(session_optimizer())

        print('Time to write those results to sheets!')
        #first optimizer space
        coords = np.array(opt.Xi).transpose().tolist()

        res = np.array(opt.yi).transpose().tolist()

        results_to_write = [[session_user_ID()]] + [['Xi']] + coords + [['res']] + [res]
        sheet_to_write = 'Sheet1'
        results_sheet.append_to_sheet(results_to_write, sheet_to_write)
        print('writing results to ', sheet_to_write)

        #now big optimizer space
        coords = np.array(opt_big.Xi).transpose().tolist()

        res = np.array(opt_big.yi).transpose().tolist()

        results_to_write = [[session_user_ID()]] + [['Xi']] + coords + [['res']] + [res]
        sheet_to_write = 'Sheet2'
        results_sheet.append_to_sheet(results_to_write, sheet_to_write)
        print('writing results to ', sheet_to_write)

        #write the parameters to sheet 3
        metadata = [[session_user_ID()]] + [[session['n_trials']]] + [session['test_order'].tolist()] + [[session['master_vol']]]+ [[session['device']]]
        sheet_to_write = 'Sheet3'
        results_sheet.append_to_sheet(metadata, sheet_to_write)
        print('writing results to ', sheet_to_write)

        return render_template("thanks.html")


    elif session['lowdim_complete'] is True:
        next_page = 'highdim'

    elif session['highdim_complete'] is True:
        next_page = 'lowdim'

    return render_template("pause.html", next_page = next_page)

def get_instructions():
    with open("static/instructions", "r") as f:
        content = f.read()
    return content





@app.route('/b', methods=['POST', 'GET'])
@app.route('/highdim', methods=['POST', 'GET'])
def highdim(dim4_params=[[1000.0, 1, 0, 0.5]]):


    noise_vol =  get_noise_vol()
    curr_noise_vols = session_noise_vols_high()
    curr_noise_vols.append(noise_vol)
    session['noise_vols_high'] = curr_noise_vols
    session.modified = True

    opt_big_space = jsonpickle.decode(session_big_optimizer())









        #coords = np.array(opt_big_space.Xi).transpose().tolist()

        #res = np.array(opt_big_space.yi).transpose().tolist()

        #results_to_write = [['Xi']] + coords + [['res']] + [res]
        #results_sheet.append_to_sheet(results_to_write, 'Sheet2')

    vals = np.squeeze(vec_to_freqpower(dim4_params).numpy()).tolist()


    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():

        if len(opt_big_space.Xi)>session_n_trials():
            print('check the optimisation, dude!')
        prev_x = opt_big_space.ask()
        #vals_previous = np.squeeze(np.abs(get_synth_params(prev_x))).tolist()
        frq_pwr_prev = get_freqs_powers([prev_x])
        vals_previous = np.squeeze(tf.concat(frq_pwr_prev, axis=1).numpy()).tolist()

        amplitudes = vals_previous[5:]
        rms_previous = calc_rms_sines(amplitudes)
        noise_val_previous = session_noise_vols_high()[-2]

        print(prev_x)


        r = form.r.data
        #f_val = -r/rms_previous

        f_val = get_adjusted_f_val( r,rms_previous, noise_val_previous)


        print('giving optimizer values from input ' + str(prev_x) + ' to yield audibility of ' + str(f_val))
        opt_big_space.tell(prev_x, f_val)
        next_x = opt_big_space.ask()
        print('next point to evaluate is ' + str(next_x))
        #vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()
        frq_pwr = get_freqs_powers([prev_x])
        vals_previous = np.squeeze(tf.concat(frq_pwr, axis=1).numpy()).tolist()

        #result1 = gp_minimize(rosenbrock, space, n_calls=1, random_state=0)
        #result1 = reload_stable(result1, addtl_calls=25, init_seed=0)

        session['big_opt'] = jsonpickle.encode(opt_big_space)
        session.modified = True

        trial_number = len(opt_big_space.Xi) + 1

        if len(opt_big_space.Xi) > session_n_trials() - 1:
            print('Alright kid, enough already!')
            session['highdim_complete'] = True
            session.modified = True
            return pause()



        else:

         return render_template(test_template, calib_vol = session_master_vol(), vals=json.dumps(vals), form=form, content=get_instructions(), slider_range=get_slider_range(), trial=trial_number, n_trials=session_n_trials(), noise_vol=noise_vol)

    else:
        next_x = [opt_big_space.ask()]
        print(next_x)

        session['big_opt'] = jsonpickle.encode(opt_big_space)
        session.modified = True


        #for some reason the lambda_fn in soiundgen_tf doesn't work here so we reshape separately
        frq_pwr = get_freqs_powers(next_x)
        vals = np.squeeze(tf.concat(frq_pwr, axis=1).numpy()).tolist()

        trial_number = len(opt_big_space.Xi) + 1
        return render_template(test_template, calib_vol = session_master_vol(), vals=json.dumps(vals), form=form, content=get_instructions(), trial=trial_number, slider_range=get_slider_range(), n_trials=session_n_trials(), noise_vol=noise_vol)




#trial_num_counter = TrialNum()
@app.route('/a', methods=['POST', 'GET'])
@app.route('/lowdim', methods=['POST', 'GET'])
def lowdim(xy=[2.0, 2.0]):


    noise_vol =  get_noise_vol()
    curr_noise_vols = session_noise_vols_low()
    curr_noise_vols.append(noise_vol)
    session['noise_vols_low'] = curr_noise_vols
    session.modified = True

    opt=jsonpickle.decode(session_optimizer())






    form = InputForm(request.form)

    #xy = [2.0, 2.0]
    #vals = np.squeeze(np.abs(get_synth_params(xy))).tolist()

    #vals = [100, 200, 300, 400, 500, 0.1, 0.1, 0.1, 0.1, 0.1]

    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():

        if len(opt.Xi)>session_n_trials():
            print('check the optimisation, dude!')
        prev_x = opt.ask()
        vals_previous = np.squeeze(np.abs(get_synth_params(prev_x))).tolist()
        amplitudes = vals_previous[5:]
        rms_previous = calc_rms_sines(amplitudes)
        noise_val_previous = session_noise_vols_low()[-2]

        print(prev_x)


        r = form.r.data
        #tone_rms = r/rms_previous
        #tone_rms = r*rms
        #the volume of the tone
        f_val = get_adjusted_f_val( r,rms_previous, noise_val_previous)
        print('giving optimizer values from input ' + str(prev_x) + ' to yield audibility of ' + str(f_val))
        opt.tell(prev_x, f_val)
        next_x = opt.ask()
        print('next point to evaluate is ' + str(next_x))
        vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()
        print("Synthing the following sines:" + str(vals))

        session['opt'] = jsonpickle.encode(opt)
        session.modified = True

        trial_number = len(opt.Xi) + 1

        if len(opt.Xi) > session_n_trials() - 1:
            print('Alright kid, enough already!')
            session['lowdim_complete'] = True
            session.modified = True
            return pause()

        else:
            return render_template(test_template, calib_vol = session_master_vol(), vals=json.dumps(vals), form=form, content=get_instructions(), slider_range=get_slider_range(), trial=trial_number, n_trials=session_n_trials(), noise_vol=noise_vol)

    else:
        next_x = opt.ask()
        session['opt'] = jsonpickle.encode(opt)
        session.modified = True


        print(next_x)
        vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()
        trial_number = len(opt.Xi) + 1

        return render_template(test_template, calib_vol = session_master_vol(), vals=json.dumps(vals), form=form, content=get_instructions(), slider_range=get_slider_range(), trial=trial_number, n_trials=session_n_trials(), noise_vol=noise_vol)

def write_validation_sheet(results):

    res_formatted = []

    for res in results:
        little_list = [res[1]] + [res[2]] + res[0]
        res_formatted.append(little_list)
    results_sheet.append_to_sheet(res_formatted, 'Sheet9')


@app.route('/validate', methods=['POST', 'GET'])
def validate():

    noise_vol=1
    all_vals_list = get_validation_vals()

    session.modified = True



    curr_validation_results = session_validation_results()
    trial_number = len(curr_validation_results)+1

    if trial_number>30:
        print ('Ey! We validated it!')
        #save results to sheet.
        write_validation_sheet(curr_validation_results)
        #results_sheet.append_to_sheet(curr_validation_results, 'Validation')
        return render_template("thanks.html")

    form = InputForm(request.form)

    if request.method == 'POST' and form.validate():


        data_previous = all_vals_list.pop()
        session['validation_vals'] = all_vals_list
        vals_previous = data_previous['values']
        amplitudes = vals_previous[5:]
        rms_previous = calc_rms_sines(amplitudes)
        noise_val_previous = 1
        print("previous vals = " + str(vals_previous))


        r = form.r.data
        #tone_rms = r/rms_previous
        #tone_rms = r*rms
        #the volume of the tone
        f_val = get_adjusted_f_val( r,rms_previous, noise_val_previous)
        print('giving optimizer values from input ' + str(vals_previous) + ' to yield audibility of ' + str(f_val))

        next_x = all_vals_list[-1]['values']
        print('next point to evaluate is ' + str(next_x))


        curr_validation_results.append([vals_previous, f_val, data_previous['type']])
        session['validation_results'] = curr_validation_results
        session.modified = True

        vals = all_vals_list[-1]['values']

        #trial_number = len(curr_validation_results)

        return render_template(test_template, calib_vol = session_master_vol(), vals=json.dumps(vals), form=form, content=get_instructions(), slider_range=get_slider_range(), trial=trial_number, n_trials=30, noise_vol=noise_vol)

    else:
        vals = all_vals_list[-1]['values']


        return render_template(test_template, calib_vol = session_master_vol(), vals=json.dumps(vals), form=form, content=get_instructions(), slider_range=get_slider_range(), trial=trial_number, n_trials=30, noise_vol=noise_vol)



@app.route('/changetone', methods=['POST', 'GET'])
def changetone(xy=[2.0, 2.0]):

    form = InputForm(request.form)

    #xy = [2.0, 2.0]
    vals = np.squeeze(np.abs(get_synth_params(xy))).tolist()

    #vals = [100, 200, 300, 400, 500, 0.1, 0.1, 0.1, 0.1, 0.1]

    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():

        if len(opt.Xi)>20:
            print('check the optimisation, dude!')
        prev_x = opt.ask()
        vals_previous = np.squeeze(np.abs(get_synth_params(prev_x))).tolist()
        amplitudes = vals_previous[5:]
        rms_previous = calc_rms_sines(amplitudes)

        print(prev_x)


        r = form.r.data
        f_val = -r/rms_previous
        print('giving optimizer values from input ' + str(prev_x) + ' to yield audibility of ' + str(f_val))
        opt.tell(prev_x, f_val)
        next_x = opt.ask()
        print('next point to evaluate is ' + str(next_x))
        vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()


        #result1 = gp_minimize(rosenbrock, space, n_calls=1, random_state=0)
        #result1 = reload_stable(result1, addtl_calls=25, init_seed=0)





        return render_template('compare_sounds_inverse.html', vals=json.dumps(vals), form=form)

    else:
        next_x = opt.ask()
        print(next_x)
        vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()

        return render_template('compare_sounds_inverse.html', vals=json.dumps(vals), form=form)


# Variant page showing 2 sliders
@app.route('/sliders', methods=['POST', 'GET'])
def sliders(xy=[2.0, 2.0]):
    form = InputForm(request.form)

    #xy = [2.0, 2.0]
    vals = np.squeeze(np.abs(get_synth_params(xy))).tolist()

    #vals = [100, 200, 300, 400, 500, 0.1, 0.1, 0.1, 0.1, 0.1]

    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():

        next_x = opt.ask()


        print(next_x)


        r = form.r.data
        f_val = r
        print('giving optimizer values from input ' + str(next_x) + ' to yield output of ' + str(r))
        opt.tell(next_x, f_val)
        next_x = opt.ask()
        print('next point to evaluate is ' + str(next_x))
        vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()

        #result1 = gp_minimize(rosenbrock, space, n_calls=1, random_state=0)
        #result1 = reload_stable(result1, addtl_calls=25, init_seed=0)





        return render_template('compare_sounds_2_sliders.html', vals=json.dumps(vals), form=form)

    else:
        next_x = opt.ask()
        print(next_x)
        vals = np.squeeze(np.abs(get_synth_params(next_x))).tolist()

        return render_template('compare_sounds_2_sliders.html', vals=json.dumps(vals), form=form)



@use_named_args(space)
def method_4_opt(**params):

    xy=[params['X'], params['Y']]
    #vals = np.squeeze(np.abs(get_synth_params(xy))).tolist()
    return index(xy)

def resty():
    form = InputForm(request.form)

    # xy = [2.0, 2.0]
    vals = np.squeeze(np.abs(get_synth_params(xy))).tolist()

    # vals = [100, 200, 300, 400, 500, 0.1, 0.1, 0.1, 0.1, 0.1]

    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        r = form.r.data
        vals_list.append(r)

        template_returned =  render_template('compare_sounds.html', vals=json.dumps(vals), form=form)
        return template_returned

    else:
        return render_template('compare_sounds.html', vals=json.dumps(vals), form=form)


@app.route('/resty', methods=['POST', 'GET'])
def mini():
    res_gp_big = gp_minimize(resty, space, n_calls=20,
                             random_state=0)

'''
@app.route('/testy', get_data())
def goog():
    template_returned = render_template('test_html.html', vals=json.dumps(vals), form=form)
    return template_returned
'''

@app.route('/start', methods=['POST', 'GET'])
def cycle(xy=[2.0, 2.0]):
    form = InputForm(request.form)
    xy = [2.0, 2.0]
    vals = np.squeeze(np.abs(get_synth_params(xy))).tolist()

    r = index(xy)
    res_gp_big = gp_minimize(method_4_opt, space, n_calls=20,
                             random_state=0)

    if request.method == 'POST' and form.validate():
        r = form.r.data
        return render_template('compare_sounds.html', vals=json.dumps(vals), form=form)



if __name__ == '__main__':
    app.run()