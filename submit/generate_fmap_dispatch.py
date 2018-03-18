import os
import time
# train1  test1:
#     JTM     ori     10 15     will be all done
#             mc      10 15     will be all done
#     JDM     ori     15
#             mc      15
#     spatial         10 15     will be all done




# os.system(  ('python generate_fmap.py --folder=spatial_10/frame --split2=True --split3=True')  )
# os.system(  ('python generate_fmap.py --folder=spatial_15/frame --split2=True --split3=True')  )
#
# os.system(  ('python generate_fmap.py --folder=JDM_ori/10 --split1=True --split2=True --split3=True')  )
# os.system(  ('python generate_fmap.py --folder=JDM_mc/10 --split1=True --split2=True --split3=True')  )
#
# os.system(  ('python generate_fmap.py --folder=JTM_ori/10 --split2=True --split3=True')  )
# os.system(  ('python generate_fmap.py --folder=JTM_mc/10 --split2=True --split3=True')  )

# while True:
#     current_time = time.localtime(time.time())
#     if current_time.tm_hour >= 5 and current_time.tm_min >= 30:
#         break
#     time.sleep(30)
#
# os.system(  ('python generate_fmap.py --folder=JTM_ori/15 --split2=True --split3=True')  )
# os.system(  ('python generate_fmap.py --folder=JTM_mc/15 --split2=True --split3=True')  )

# os.system(  ('python generate_fmap.py --folder=JDM_ori/15 --split2=True --split3=True')  )
# os.system(  ('python generate_fmap.py --folder=JDM_mc/15 --split2=True --split3=True')  )


os.system(  ('python generate_fmap.py --folder=spatial_10/frame --split1=True --model=vgg19 --layer=block4_pool')  )
