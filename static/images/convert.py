from PIL import Image

# 打开原始图像
img = Image.open('stanford.png')

# 将图像转换为RGBA模式，以便处理透明度
img = img.convert("RGBA")

# 获取图像的像素数据
data = img.getdata()

# 新的像素数据
new_data = []

for item in data:
    # 判断是否为白色、灰色或接近白色/灰色的区域
    # 白色和灰色的RGB值都在较高范围内
    r, g, b, a = item
    if r in range(200, 256) and g in range(200, 256) and b in range(200, 256):
        new_data.append((255, 255, 255, 0))  # 白色和灰色变为透明
    else:
        new_data.append(item)  # 保留其他颜色

# 更新图像数据
img.putdata(new_data)

# 保存修改后的图像
img.save('stanford2.png')
