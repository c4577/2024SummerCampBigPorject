# 图片风格迁移 README

## 一、项目简介

### 1、选题依据
风格迁移（Style Transfer）是一种将艺术风格应用到图片内容上的技术，广泛应用于图像处理和计算机视觉领域。这项技术使得普通图像能够呈现出类似著名艺术家的风格，如梵高的“星夜”或毕加索的立体主义风格。传统的风格迁移方法通常依赖于手工设计的特征提取和优化算法，且效果受限于特定的场景和风格。随着深度学习的发展，基于卷积神经网络（CNN）的风格迁移方法成为主流，其通过深层网络提取更为细致的图像特征。

本项目的目标是利用VGG19模型进行图像的风格迁移，实现自然且细腻的风格迁移效果。这一研究不仅对艺术创作和图像处理领域有重要意义，还能促进相关技术在实际应用中的发展和优化。项目使用VGG19网络，将风格图像的风格应用于内容图像，实现了基于风格迁移的图像处理。风格迁移技术可以将一种图像的艺术风格转移到另一种图像中，同时保留内容图像的主要特征。

### 2、业界现状
当前，风格迁移技术主要依赖于卷积神经网络（CNN）进行特征提取。VGG19模型作为一种经典的深度学习模型，已经被广泛应用于风格迁移中。现有的解决方案包括但不限于神经风格迁移（Neural Style Transfer）和生成对抗网络（GANs）等方法。这些方法通常通过计算内容和风格的损失函数来优化图像。

### 3、本项目介绍
本项目设计包括以下几个主要模块：

特征提取模块：基于VGG19模型提取图像的内容特征和风格特征。

损失计算模块：计算内容损失、风格损失和总变差损失，用于优化生成图像。计算内容损失：使用VGG19模型对内容图像和生成图像进行前向传播，提取指定内容层的特征图，使用函数计算这些特征图之间的均方误差，度量生成图像与内容图像在结构上的差异。计算风格损失：使用函数计算风格图像和生成图像在指定风格层的Gram矩阵。Gram矩阵通过计算特征图的内积来描述风格，通过比较生成图像的Gram矩阵与目标图像的Gram矩阵，计算均方误差。这度量了生成图像与风格图像在风格上的差异。计算总变差损失：计算生成图像中相邻像素的差异，度量图像的平滑性。它通过计算x轴和y轴方向的像素差异平方和来实现，有助于减少图像中的噪声和不自然的纹理。

图像优化模块：通过梯度下降优化算法调整生成图像，多次迭代使其逐渐接近目标风格。



## 二、依赖

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

## 三、文件说明

- `content.jpg`：内容图像，用于定义最终图像的主要结构。
- `style.jpg`：风格图像，用于定义最终图像的艺术风格。
- `output_image.jpg`：最终生成的风格迁移图像。
- `output_image_{epoch}.jpg`：每个训练轮次保存的中间结果图像。

## 四、使用说明

### 1. 环境设置

确保已安装所需的Python库。可以使用以下命令安装：
```bash
pip install tensorflow numpy opencv-python matplotlib
```

### 2. 代码说明

- **`load_and_process_image(image_path, target_size=(512, 512))`**：加载并预处理图像。
- **`deprocess_image(image)`**：将生成图像从[-1,1]范围转换回[0,255]范围。
- **`save_image(image, filename)`**：保存生成的图像。
- **`load_vgg19_model()`**：加载VGG19模型并提取内容和风格层。
- **`gram_matrix(tensor)`**：计算Gram矩阵，用于风格损失计算。
- **`compute_content_loss(base_content, target)`**：计算内容损失。
- **`compute_style_loss(base_style, target)`**：计算风格损失。
- **`total_variation_loss(image)`**：计算总变差损失以平滑图像。
- **`train_step(image, model, content_features, style_features, content_weight, style_weight, total_variation_weight)`**：训练步骤，包括损失计算和梯度更新。

### 3. 运行示例

在代码文件中设置图像路径并运行脚本：
```python
# 加载图像
content_image = load_and_process_image('content.jpg')
style_image = load_and_process_image('style.jpg')

# 获取内容和风格特征
model, content_layers, style_layers = load_vgg19_model()
content_features = model(content_image)[0]
style_features = [gram_matrix(layer) for layer in model(style_image)[1:]]

# 初始化生成图像
generated_image = tf.Variable(content_image, dtype=tf.float32)

# 设置超参数
content_weight = 1e4
style_weight = 1e-2
total_variation_weight = 1e-6

optimizer = tf.optimizers.Adam(learning_rate=0.001)

# 训练风格迁移模型
num_epochs = 10
for epoch in range(num_epochs):
    loss = train_step(generated_image, model, content_features, style_features, content_weight, style_weight, total_variation_weight)
    if epoch % 2 == 0:
        save_image(generated_image, f'output_image_{epoch}.jpg')
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# 保存最终生成图像
save_image(generated_image, 'output_image.jpg')

# 显示最终生成图像
plt.imshow(deprocess_image(generated_image.numpy()))
plt.axis('off')
plt.show()
```

### 4. 调整参数

- **`content_weight`**：控制内容保留的程度。
- **`style_weight`**：控制风格迁移的程度。
- **`total_variation_weight`**：控制图像平滑度。


## 五、项目测试
### 测试一：
风格图像：![image](https://github.com/user-attachments/assets/910b9ae2-49b8-4a26-a29a-cd0dc043deb6)

内容图像：![image](https://github.com/user-attachments/assets/0b8b19da-b299-475a-a7fb-1906397b641c)

结果图像：![image](https://github.com/user-attachments/assets/febb3fcd-b7f7-4dc3-874b-5e1e404c9ed3)

 ### 测试二：
风格图像：![image](https://github.com/user-attachments/assets/2b421f1f-e173-4a45-a02a-27a24008822d)

内容图像：![image](https://github.com/user-attachments/assets/368b3900-7022-4366-a000-cf53010c5acc)

结果图像：![image](https://github.com/user-attachments/assets/50e62478-34ef-4797-8638-ac9878f16b68)

 
## 六、项目管理
任务分工

迟忆雯：统筹管理，项目调研，程序调试

胡梦洋：项目调研，程序编写，程序调试

岳文惠：项目调研，程序调试，汇报PPT制作

李舒涵：项目调研，程序调试，项目汇报

每一位同学都及时完成了各自的任务，沟通高效，合作顺利，为项目最后的完成都有着很大的贡献。

## 七、总结与反思
从技术层面，我们对一些比较先进的技术有了初步了解、学习，并加以探索、运用。在团队合作方面，初次尝试了利用github进行项目管理，并提高了任务分配和解决复杂问题的能力，强化了责任意识，同时增进了同学之间的友谊。
