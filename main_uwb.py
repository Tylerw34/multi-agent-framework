from class_def.locator import UWBLocator, UWBLocator_ROS

locator = UWBLocator(1, 1, mode="TAG")
locator.run()
# 检测键盘退出
while True:
    if input() == 'q':
        locator.stop()
        break
