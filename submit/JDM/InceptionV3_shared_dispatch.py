import os

e_begin = 0

for i in range(10):
    model_name = 'e' + str(e_begin) + '_spe10_round2.h5'
    if i == 0:
        os.system(('python InceptionV3_shared.py --echo_begin=' + str(i) + ' --echo_end=' + str(i+1)))
        os.system(('python Evaluate_InceptionV3_shared.py --model_name=' + model_name))
    else:
        os.system(('python InceptionV3_shared.py --echo_begin=' + str(i) + ' --echo_end=' + str(i + 1)) + ' --first=False --model_name=' + model_name)
        e_begin += 1
        new_model_name = 'e' + str(e_begin) + '_spe10_round2.h5'
        os.system(('python Evaluate_InceptionV3_shared.py --model_name=' + new_model_name))


