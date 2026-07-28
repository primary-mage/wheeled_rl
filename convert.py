import mujoco
# 1. 读取我们刚修好路径的 URDF
model = mujoco.MjModel.from_xml_path('./asset/urdf_1534/Total.SLDASM/urdf/Total.SLDASM.urdf')
# 2. 把它保存为 MuJoCo 原生的 XML 格式
mujoco.mj_saveLastXML('./asset/urdf_1534/Total.SLDASM/urdf/wheeled_robot.xml', model)
print("转换成功！")
