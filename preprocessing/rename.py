import os
import sys

c = int(sys.argv[3])
im_pths = [f for dp, dn, filenames in os.walk("/Users/aishasiddiqa/Desktop") for f in filenames if os.path.splitext(f)[1] == sys.argv[2]]
for i in im_pths:
    dst = sys.argv[1] + str(c) + ".png"
    os.rename(i,dst)
    c += 1

#argv[1] name of the images
#argv[2] file extension png jpeg jpg of the source files
#argv[3] number to start from naming the images
