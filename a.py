with open("/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/a.txt") as f:
    res = f.readlines()
res = [i.split(":") for i in res]
for i in range(len(res)):
    res[i][0] = int(res[i][0][1: -1])
    res[i][1] = "wisdota/" + res[i][1].split("/")[-1][:-3]
for i in res:
    print(i[1], i[0])