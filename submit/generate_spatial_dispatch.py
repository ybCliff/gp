import os


os.system(  ('python generate_spatial.py --frame=15 --scope=' + 'train2')  )
os.system(  ('python generate_spatial.py --frame=15 --scope=' + 'test2')  )
os.system(  ('python generate_spatial.py --frame=10 --scope=' + 'train2')  )
os.system(  ('python generate_spatial.py --frame=10 --scope=' + 'test2')  )

os.system(  ('python generate_spatial.py --frame=15 --scope=' + 'train3')  )
os.system(  ('python generate_spatial.py --frame=15 --scope=' + 'test3')  )
os.system(  ('python generate_spatial.py --frame=10 --scope=' + 'train3')  )
os.system(  ('python generate_spatial.py --frame=10 --scope=' + 'test3')  )