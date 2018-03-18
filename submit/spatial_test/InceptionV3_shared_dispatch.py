import os

# os.system(('python InceptionV3_shared.py --echo_begin=0 --echo_end=1 --first=1 --learning_rate=0.01 --spe=4 --momentum=0 --split=1'))
# os.system(('python InceptionV3_shared.py --echo_begin=1 --echo_end=2 --first=0 --model_name=e0_spe4_round2.h5 --learning_rate=0.01 --spe=4 --momentum=0 --split=1'))
# os.system(('python InceptionV3_shared.py --echo_begin=2 --echo_end=3 --first=0 --model_name=e1_spe4_round2.h5 --learning_rate=0.001 --spe=4 --momentum=0.8 --split=1'))

# os.system(('python Evaluate_InceptionV3_shared.py --split=1 --model_name=e0_spe4_round2.h5'))
# os.system(('python Evaluate_InceptionV3_shared.py --split=1 --model_name=e1_spe4_round2.h5'))
# os.system(('python Evaluate_InceptionV3_shared.py --split=1 --model_name=e2_spe4_round2.h5'))

os.system(('python generate_InceptionV3_fmap.py --split=1 --model_name=e2_spe4_round2.h5 --layer=fc1'))
os.system(('python generate_InceptionV3_fmap.py --split=1 --model_name=e2_spe4_round2.h5 --layer=fc2'))

