import os

# os.system(('python InceptionV3_shared.py --echo_begin=0 --echo_end=1 --first=1 --learning_rate=0.01 --spe=4 --momentum=0.5'))
# os.system(('python InceptionV3_shared.py --echo_begin=2 --echo_end=3 --first=0 --learning_rate=0.0001 --spe=4 --momentum=0.9 --model_name=e1_spe4_round2.h5'))
#
# os.system(('python Evaluate_InceptionV3_shared.py --model_name=e0_spe4_round2.h5'))
os.system(('python Evaluate_InceptionV3_shared.py --model_name=e2_spe4_round2.h5'))

# os.system(('python generate_InceptionV3_fmap.py --model_name=e1_spe4_round2.h5 --layer=fc1'))
# os.system(('python generate_InceptionV3_fmap.py --model_name=e1_spe4_round2.h5 --layer=fc2'))


# os.system(('python Evaluate_InceptionV3_shared.py --model_name=e1_spe4_round2.h5'))
# for i in range(10):
#     model_name = 'e' + str(e_begin) + '_spe10_round2.h5'
#     if i == 0:
#         os.system(('python InceptionV3_shared.py --echo_begin=' + str(i) + ' --echo_end=' + str(i+1)))
#         os.system(('python Evaluate_InceptionV3_shared.py --model_name=' + model_name))
#     else:
#         os.system(('python InceptionV3_shared.py --echo_begin=' + str(i) + ' --echo_end=' + str(i + 1)) + ' --first=False --model_name=' + model_name)
#         e_begin += 1
#         new_model_name = 'e' + str(e_begin) + '_spe10_round2.h5'
#         os.system(('python Evaluate_InceptionV3_shared.py --model_name=' + new_model_name))

# model_name = 'e' + str(e_begin) + '_spe10_round2.h5'
# os.system(('python Evaluate_InceptionV3_shared.py --model_name=' + model_name))
