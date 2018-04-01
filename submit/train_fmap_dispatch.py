import os
N = 3
for i in range(N):
    os.system('python train_fmap2.py --version='+str(i)+' --folder=JDM_ori/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg19 --layer=block5_pool')
for i in range(N):
    os.system('python train_fmap2.py --version='+str(i)+' --folder=JDM_ori/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg19 --layer=block4_pool')
for i in range(N):
    os.system('python train_fmap2.py --version='+str(i)+' --folder=JDM_ori/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg16 --layer=block5_pool')
for i in range(N):
    os.system('python train_fmap2.py --version='+str(i)+' --folder=JDM_ori/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg16 --layer=block4_pool')

for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_mc/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg19 --layer=block5_pool')
for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_mc/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg19 --layer=block4_pool')
for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_mc/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg16 --layer=block5_pool')
for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_mc/10 --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg16 --layer=block4_pool')

for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_ori/10 --folder2=JDM_mc/10 --read_type=mean --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg19 --layer=block5_pool')
for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_ori/10 --folder2=JDM_mc/10 --read_type=mean --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg19 --layer=block4_pool')
for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_ori/10 --folder2=JDM_mc/10 --read_type=mean --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg16 --layer=block5_pool')
for i in range(N):
    os.system('python train_fmap2.py --version=' + str(i) + ' --folder=JDM_ori/10 --folder2=JDM_mc/10 --read_type=mean --acc_limit=0.23 --csv_path=Evaluate_JDM/ --model=vgg16 --layer=block4_pool')


os.system(  ('python generate_fmap.py --folder=JDM_ori/10_all --split1=True --model=vgg19 --layer=block5_pool')  )
os.system(  ('python generate_fmap.py --folder=JDM_mc/10_all --split1=True --model=vgg19 --layer=block5_pool')  )
os.system(  ('python generate_fmap.py --folder=JDM_ori/10_all --split1=True --model=vgg16 --layer=block5_pool')  )
os.system(  ('python generate_fmap.py --folder=JDM_mc/10_all --split1=True --model=vgg16 --layer=block5_pool')  )
