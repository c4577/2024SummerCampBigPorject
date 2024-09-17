import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image

# 加载图片并预处理
def load_and_process_img(img_path, max_dim=512):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = np.expand_dims(img, axis=0)
    img = img.copy()  # 确保 img 是可写的
    img = preprocess_input(img)
    return img

# 反向处理图片
def deprocess_img(processed_img):
    processed_img = processed_img.copy()
    processed_img = processed_img.reshape((processed_img.shape[1], processed_img.shape[2], 3))
    processed_img += [103.939, 116.779, 123.68]
    processed_img = processed_img[:, :, ::-1]
    processed_img = np.clip(processed_img, 0, 255).astype('uint8')
    return processed_img

# 计算内容损失
def content_loss(base_content, target):
    return tf.reduce_sum(tf.square(base_content - target))

# 计算 Gram 矩阵
def gram_matrix(x):
    x = tf.squeeze(x)  # 去掉批次维度，形状为 (height, width, channels)
    height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    
    # 将 (height, width, channels) 重塑为 (height * width, channels)
    x = tf.reshape(x, [height * width, channels])
    
    # 计算 Gram 矩阵
    gram = tf.matmul(x, x, transpose_a=True)
    
    # 归一化 Gram 矩阵
    num_elements = tf.cast(height * width * channels, tf.float32)
    gram /= num_elements
    
    return gram

# 计算风格损失
def style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_sum(tf.square(gram_style - gram_target))

# 定义风格迁移模型
def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    model = tf.keras.models.Model([vgg.input], outputs)
    return model

# 风格迁移过程
def style_transfer(content_path, style_path, epochs=10, steps_per_epoch=100):
    model = get_model()
    content_img = load_and_process_img(content_path)
    style_img = load_and_process_img(style_path)
    content_features = model(content_img)
    style_features = model(style_img)

    content_feature = content_features[0]
    style_features_list = style_features[1:]
    style_grams = [gram_matrix(style_feature) for style_feature in style_features_list]

    # 初始化生成图像
    generated_img = tf.Variable(content_img, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            with tf.GradientTape() as tape:
                generated_features = model(generated_img)
                gen_content_feature = generated_features[0]
                gen_style_features = generated_features[1:]
                
                c_loss = content_loss(gen_content_feature, content_feature)
                s_loss = tf.add_n([style_loss(gen_style, style_gram) for gen_style, style_gram in zip(gen_style_features, style_grams)])
                total_loss = c_loss + s_loss
                
            grads = tape.gradient(total_loss, generated_img)
            optimizer.apply_gradients([(grads, generated_img)])
            
            # 打印损失信息
            if step % 100 == 0:
                print(f'Epoch {epoch + 1}, Step {step + 1}, Loss: {total_loss.numpy()}')

        # 每 2 个 epoch 保存一次图像
        if (epoch + 1) % 2 == 0:
            result_img = deprocess_img(generated_img.numpy())
            output_path = f'C:/Users/HUMENGYANG/output_epoch_{epoch+1}.jpg'
            Image.fromarray(result_img).save(output_path)
            print(f"Saved image for epoch {epoch+1} at {output_path}")

    final_img = generated_img.numpy()
    return deprocess_img(final_img)

# 执行风格迁移
content_path = 'C:/Users/HUMENGYANG/content.jpg'
style_path = 'C:/Users/HUMENGYANG/style.jpg'
result_img = style_transfer(content_path, style_path, epochs=20)  # 设置 epochs 为 10

# 最终结果图像保存
output_path = 'C:/Users/HUMENGYANG/output_final.jpg'
Image.fromarray(result_img).save(output_path)

# 显示最终结果
plt.imshow(result_img)
plt.axis('off')
plt.show()
